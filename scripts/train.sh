
for len in 3 6 12
do
for num in 4 1 7
do
for rate in 0.0005 0.0001 
do
python -u main.py \
  --model_name 'STD2Vformer'\
  --data_name 'PEMS04' \
  --lr $rate \
  --M $num\
  --pred_len $len\
  --batch_size 32 \
  --resume_dir None\
  --output_dir None\
  --info 'None'\

done
done
done

for len in 3 6 12
do
for num in 6 3 10
do
for rate in 0.0005 0.0001 
do
python -u main.py \
  --model_name 'STD2Vformer'\
  --data_name 'PEMS08' \
  --lr $rate \
  --M $num\
  --pred_len $len\
  --batch_size 32 \
  --resume_dir None\
  --output_dir None\
  --info 'None'\

done
done
done

for len in 3 6 12
do
for num in 2 8 9
do
for rate in 0.0005 0.0001 
do
python -u main.py \
  --model_name 'STD2Vformer'\
  --data_name 'METR-LA' \
  --lr $rate \
  --M $num\
  --pred_len $len\
  --batch_size 16 \
  --resume_dir None\
  --output_dir None\
  --info 'None'\

done
done
done