@setlocal enabledelayedexpansion
@set train_root="E:\dataset\cifar\train"
@set test_root="E:\dataset\cifar\test"
start python im2rec.py --list  --recursive fortrain !train_root!\
start python im2rec.py --list  --recursive fortest !test_root!\
@echo waiting for list done
pause
start python im2rec.py --num-thread 1 --pass-through  fortrain !train_root!\
start python im2rec.py --num-thread 1 --pass-through  fortest !test_root!\
@endlocal
@pause