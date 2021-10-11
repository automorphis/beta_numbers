# Beta Expansions of Salem Numbers

## About

This project calculates the [beta expansions](https://en.wikipedia.org/wiki/Non-integer_base_of_numeration) of [Salem numbers](https://en.wikipedia.org/wiki/Salem_number). 

Schmidt [proved](https://londmathsoc.onlinelibrary.wiley.com/doi/abs/10.1112/blms/12.4.269) that all rationals in [0,1] have eventually periodic expansions with respect to [Pisot numbers](https://en.wikipedia.org/wiki/Pisot%E2%80%93Vijayaraghavan_number) and conjectures the same for Salem. Boyd [proved](https://www.degruyter.com/document/doi/10.1515/9783110852790.57/html) that 1 has an eventually periodic beta expansion with respect to a degree four Salem number. Later, Boyd [calculated](https://www.ams.org/journals/mcom/1996-65-214/S0025-5718-96-00700-4/S0025-5718-96-00700-4.pdf) the Beta expansions of almost all Salem numbers of degree six and trace at most 15. In the same paper, he provided a heuristic (but not formally rigorous) argument suggesting that the beta expansion of 1 with respect to any Salem number of degree six is eventually periodic, but the same is not true of 8 or any higher degree.  

This project has two aims:

1. Continue the calculation for degree six where Boyd left off. The hope is that this will provide empirical evidence that will suggest how to refine his heuristic argument, with the goal of making it formally rigorous.
2. Program an intuitive and user-friendly interface for working with data saved on the disk and in RAM simultaneously, especially when the data is the orbit of a point under a fixed transformation. For example, if the user is calculating a very long orbit of a point `initial_point` under a transformation `T`, then they can request the `n`-th iterate from the disk or RAM, wherever it may be, simply through the call `register.get_n(n, T, initial_point)`.

## Calculations so far

Each iterate takes ~ 85 microseconds on my machine. 

| Salem number       | Minimal polynomial              | Number of calculated coefficients | Periodicity |
| -------------------|-------------------------------- | --------------------------------- | ----------- |
| 13.345638433018787 | (1, -10, -40, -59, -40, -10, 1) | 2 billion                         | unknown     |

## To do

### Improvements

1. ~~Have `Pickle_Register` dynamical discover `Save_State`s.~~
2. ~~Edit `calc_orbit.calc_period_ram_and_disk` to pick up where a previous calculation left off.~~
3. Implement multiprocessing for calculating orbits.
4. ~~Cythonize `beta_orbit.Beta_Orbit_Iter`.~~
5. Publish a package for this project.
6. Publish a package for `Int_Polynomial`.
7. Cythonize `mpmath.polyroots`
8. Create a new, faster, more portable `Register`. Currently looking at LevelDB and the Plyvel module.

### Known bugs

1. ~~.`calc_orbit.calc_period_ram_and_disk` still has a few bugs~~
2. Repeatedly instantiating and dereferencing moderately sized NumPy arrays (~100,000 entries) with `dtype = object` will sometimes cause Sage to hang. I do not know why this behavior occurs. I have not been able to localize it, nor do I know if it is unique to Sage or if it also occurs in a standard Python virtualenv. [This AskSage question](https://ask.sagemath.org/question/53245/why-is-numpy-slower-inside-of-a-sage-notebook/) could provide some insight. Best practice is to just avoid using NumPy arrays with `dtype = object` altogether; either create an extension type (as in the case of `Int_Polynomial_Array`) or use a `list`.
