import datetime
import multiprocessing
import time
from pathlib import Path
import itertools

from cornifer import load
from cornifer._utilities import BYTES_PER_MB
from intpolynomials import Int_Polynomial_Register, Int_Polynomial


def calc_perron_polys(worker, numWorkers, timeout, poly_reg, beta0_reg, max_perron_polys_per_deg, save_period):

    import logging
    import time
    from pathlib import Path
    import itertools
    import traceback

    import psutil
    from mpmath import mp, nstr, inf
    from cornifer import Apri_Info, Data_Not_Found_Error, Block, Apos_Info
    from intpolynomials import Int_Polynomial_Iter, Int_Polynomial_Array

    from beta_numbers.perron_numbers import Perron_Number, Not_Perron_Error

    call_start = time.time()
    p = psutil.Process()
    p.cpu_affinity([worker])
    time.sleep(1 + 3 * worker)
    log_filename = Path(__file__).parent / ".." / "logs" / f"perron_numbers_log{worker}.txt"
    log_filename = log_filename.resolve()
    logging.basicConfig(filename = log_filename, level = logging.INFO)
    logging.info(f"Spawned worker process and logging. ({str(datetime.datetime.now())})")
    perron_polys_this_call = 0

    try:

        mp.dps = 80
        sigfigs = mp.dps - 10

        with poly_reg.open() as poly_reg:

            with beta0_reg.open() as beta0_reg:

                for deg in itertools.count(1 + worker, numWorkers):

                    if worker == 3:

                        deg = 2
                        max_perron_polys_per_deg = 10 ** 8

                    deg_start = time.time()

                    if worker != 3 and deg <= 2:
                        continue

                    perron_polys_this_deg = 0

                    for s in itertools.count(1):

                        min_poly_apri = Apri_Info(
                            deg = deg,
                            sum_abs_coef = s,
                            mp_dps = mp.dps
                        )

                        beta0_apri = Apri_Info(
                            respective = min_poly_apri
                        )

                        if min_poly_apri in poly_reg:

                            perron_polys_this_s = poly_reg.get_total_length(min_poly_apri)
                            perron_polys_this_deg += perron_polys_this_s

                            if perron_polys_this_deg >= max_perron_polys_per_deg:
                                break # s loop

                            try:

                                if not poly_reg.get_apos_info(min_poly_apri).complete:
                                    partial = True

                                else:
                                    partial = False

                            except (Data_Not_Found_Error, AttributeError):
                                partial = True

                            if partial:

                                try:
                                    last_poly = poly_reg.get_apos_info(min_poly_apri).last_poly

                                except (Data_Not_Found_Error, AttributeError):

                                    length = poly_reg.get_total_length(min_poly_apri)

                                    if length == 0:

                                        it = Int_Polynomial_Iter(deg, s, True)
                                        partial = False

                                    else:

                                        last_poly = poly_reg[min_poly_apri, length - 1]
                                        it = Int_Polynomial_Iter(deg, s, True, last_poly)

                                else:

                                    apos = poly_reg.get_apos_info(min_poly_apri)
                                    del apos.last_poly
                                    poly_reg.set_apos_info(min_poly_apri, apos)
                                    last_poly = Int_Polynomial(deg).set(last_poly)
                                    it = Int_Polynomial_Iter(deg, s, True, last_poly)

                            else:
                                continue # s loop

                        else:

                            perron_polys_this_s = 0
                            partial = False
                            it = Int_Polynomial_Iter(deg, s, True)

                        logging.info(f"Calculating perron polys with deg = {deg} and s = {s}....")
                        logging.info(f"Found {perron_polys_this_deg} this deg and {perron_polys_this_s} this s.")

                        if partial:

                            log_first_poly = True
                            logging.info(f"Found partial set of polynomials, last_poly is {str(last_poly)}.")

                        else:
                            log_first_poly = False

                        logging.info(f"Time since beginning of deg loop: {(time.time() - deg_start)}")

                        s_start = time.time()
                        min_poly_seg = Int_Polynomial_Array(deg)
                        min_poly_seg.empty(save_period)
                        min_poly_blk = Block(min_poly_seg, min_poly_apri)
                        beta0_seg = []
                        beta0_blk = Block(beta0_seg, beta0_apri)

                        for i, p in enumerate(it):

                            if perron_polys_this_deg >= max_perron_polys_per_deg:

                                if len(min_poly_blk) > 0:

                                    poly_reg.append_disk_block(min_poly_blk)
                                    beta0_reg.append_disk_block(beta0_blk)

                                poly_reg.set_apos_info(min_poly_apri, Apos_Info(
                                    complete = False,
                                    last_poly = tuple([int(x) for x in p.get_ndarray()])
                                ))
                                logging.info("Reached maximum number of polys this deg.")
                                break # `it` loop

                            if time.time() - call_start >= timeout:

                                if len(min_poly_blk) > 0:

                                    poly_reg.append_disk_block(min_poly_blk)
                                    beta0_reg.append_disk_block(beta0_blk)

                                poly_reg.set_apos_info(min_poly_apri, Apos_Info(
                                    complete = False,
                                    last_poly = tuple([int(x) for x in p.get_ndarray()])
                                ))
                                logging.info(
                                    f"Timedout. "
                                    f"{(perron_polys_this_call/(time.time() - call_start)):.2f} perron polys per sec this call. "
                                    f"({str(datetime.datetime.now())})"
                                )
                                return

                            if log_first_poly:

                                logging.info(f"Restarting from {str(p)}...")
                                log_first_poly = False

                            if p.is_irreducible():

                                perron = Perron_Number(p)

                                try:
                                    beta0, _ = perron.calc_roots()

                                except Not_Perron_Error:
                                    pass

                                else:

                                    beta0 = nstr(beta0, sigfigs, strip_zeros=False, min_fixed=-inf, max_fixed=inf)
                                    perron_polys_this_s += 1
                                    perron_polys_this_deg += 1
                                    perron_polys_this_call += 1
                                    min_poly_seg.append(p)
                                    beta0_seg.append(beta0)

                                if len(min_poly_blk) == save_period:

                                    poly_reg.append_disk_block(min_poly_blk)
                                    beta0_reg.append_disk_block(beta0_blk)
                                    min_poly_seg.clear()
                                    beta0_seg.clear()

                        else:

                            if len(min_poly_blk) > 0:

                                poly_reg.append_disk_block(min_poly_blk)
                                beta0_reg.append_disk_block(beta0_blk)

                            logging.info(
                                f"Found {perron_polys_this_s} Perron polys this s and {perron_polys_this_deg} this deg. "
                                f"({(time.time() - s_start):.3f} sec)"
                            )

                            poly_reg.set_apos_info(min_poly_apri, Apos_Info(complete=True))
                            continue # s loop

                        break # s loop, only reached if the `it` loop breaks

    except BaseException:

        logging.critical(f"Exception raised : {traceback.format_exc()}")
        logging.critical("Returning.")
        logging.critical(f"Poly reg : {str(poly_reg._local_dir)}")
        logging.critical(f"beta0 reg : {str(beta0_reg._local_dir)}")
        return

if __name__ == "__main__":

    numWorkers = 4
    max_perron_polys_per_deg = 4 * 10 ** 7
    timeout = 20 * 60
    save_period = 1000

    exclude_timeout = int(1.8 * timeout)

    # saves_directory = Path("D:/perron_numbers")
    saves_directory = Path("/mnt/d/perron_numbers")

    poly_regs = [
        load(saves_directory / "akxh"),
        load(saves_directory / "fcAH"),
        load(saves_directory / "TfBx"),
        load(saves_directory / "6KzR")
    ]

    beta0_regs = [
        load(saves_directory / "sZxT"),
        load(saves_directory / "wT3L"),
        load(saves_directory / "jfzR"),
        load(saves_directory / "8UDU")
    ]

    # for reg in poly_regs + beta0_regs:
    #
    #     with reg.open() as reg:
    #
    #         print(reg._db_map_size, str(reg))
    #
    #         reg.increase_register_size(50 * BYTES_PER_MB)
    #
    #         print(reg._db_map_size, str(reg))

    exclude_iter = iter(itertools.cycle(range(numWorkers)))
    exclude_start = None

    while True:

        if exclude_start is None or time.time() - exclude_start >= exclude_timeout:

            exclude = 3 #next(exclude_iter)
            workers = list(range(numWorkers))
            workers.remove(exclude)
            exclude_start = time.time()

        print(workers)

        with multiprocessing.Pool(processes = numWorkers - 1) as pool:

            for worker in workers:
                x = pool.apply_async(calc_perron_polys, (worker, numWorkers, timeout, poly_regs[worker], beta0_regs[worker], max_perron_polys_per_deg, save_period))

            pool.close()
            pool.join()

        time.sleep(20)