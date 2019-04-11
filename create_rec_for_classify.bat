@setlocal enabledelayedexpansion
@set train_root=train
@set test_root=test
python im2rec.py --list  --recursive fortrain !train_root!\
python im2rec.py --list  --recursive --test-ratio 1.0 --train-ratio 0.0 fortest !test_root!\

python im2rec.py --num-thread 2 --pass-through  fortrain !train_root!\
python im2rec.py --num-thread 2 --pass-through  fortest !test_root!\
@endlocal
@pause