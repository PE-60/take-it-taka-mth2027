



# More than a Hack 2027 — Phase 1: Ball Possession

**Organised by [Metrica Sports](https://metrica-sports.com), [FC Barcelona](https://www.fcbarcelona.com)
and the [Mobile World Capital Barcelona](https://www.mwcbarcelona.com)**, at the Talent Arena in Barcelona.

A football data challenge on real professional match data, open to analysts, data scientists and
engineers. You are given one complete LaLiga match — two camera feeds, automatic tracking of every
player and the ball, automatic tactical tagging, and the camera-to-pitch geometry — and one
question to answer with it.

**Original repository of the challenge can be found [here](https://github.com/metrica-sports/more-than-a-hack-2027).**



# Team Take-It-Taka Solution to Phase 1

Documentation on our approach to solving the Initial Challange of labeling possession for each frame can be found [here](Take-It-Taka.pdf).

**In order to reproduce our final delivarable** csv file with the possession labeling of the whole tactical video, run in the root folder of the repo:
```
python run.py
```
The script informs you what it does at each step. **The final file is ```evaluation/results/submission.csv```**.




## Licence

The code and documentation in this repository are released under the MIT Licence — see
[LICENSE](LICENSE). **The match data is not covered by it**: it is separately licensed for this
competition only, under the enrollment terms of the competition.

