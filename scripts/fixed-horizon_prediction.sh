for len in 3 6 12
do
for num in 9
do
for rate in 0.0005 0.0001 
do
python -u main.py \
  --model_name 'STD2Vformer'\
  --data_name 'METR-LA' \
  --points_per_hour 12 \
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


for len in 3 6 12
do
for num in 6
do
for rate in 0.0005 0.0001 
do
python -u main.py \
  --model_name 'STD2Vformer'\
  --data_name 'PEMS08' \
  --points_per_hour 12 \
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
for num in 4 7
do
for rate in 0.0005 0.0001 
do
python -u main.py \
  --model_name 'STD2Vformer'\
  --data_name 'PEMS04' \
  --points_per_hour 12 \
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
