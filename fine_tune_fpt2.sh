export TRAIN_FILE=/home/ddemeter/CS-497/woz.train_b.txt
export TEST_FILE=/home/ddemeter/CS-497/woz.test_b.txt
python run_clm.py --output_dir=/home/ddemeter/CS-497/b --model_type=gpt2 --model_name_or_path=gpt2 --train_file=$TRAIN_FILE --do_train --validation_file=$TEST_FILE --do_eval --save_total_limit 10 --num_train_epochs 10
