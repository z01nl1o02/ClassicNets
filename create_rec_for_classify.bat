@setlocal enabledelayedexpansion
@set train_root="train"
@set test_root="test"
start python im2rec.py --list  --recursive fortrain !train_root!\
start python im2rec.py --list  --recursive fortest !test_root!\
pause
start python im2rec.py --num-thread 1 --pass-through  fortrain !train_root!\
start python im2rec.py --num-thread 1 --pass-through  fortest !test_root!\
@endlocal
@pause