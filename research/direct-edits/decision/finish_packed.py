import time
from fetch import ROOT
wd=ROOT/'weights/ov-GPU-f32/packed-ordered/whole16'
deadline=time.monotonic()+5400
while not (wd/'paired.json').exists():
 if time.monotonic()>deadline:raise TimeoutError('Full ordered primary predictions have not completed')
 time.sleep(8)
from prior import run as pr
from trust import run as tr
from discriminant import evaluate as disc
from complement import evaluate as fusion
from report import create
for fn in (pr,tr):fn('score','ov-GPU-f32','packed-ordered','whole16')
disc()
fusion()
create()
