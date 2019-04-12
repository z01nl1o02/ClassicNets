@setlocal enabledelayedexpansion
@set train_root="train"
@set test_root="test"
python im2rec.py --list  --recursive fortrain !train_root!\
python im2rec.py --list  --recursive fortest !test_root!\

python im2rec.py --num-thread 4 --pass-through  fortrain !train_root!\
python im2rec.py --num-thread 4 --pass-through  fortest !test_root!\
@endlocal
@pause