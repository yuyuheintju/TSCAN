### In-hospital mortality prediction

Run the following command to train the neural network which gives the best result. We got the best performance on validation set after 28 epochs.
       
       python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/split_48.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --output_dir mimic3models/in_hospital_mortality

### Decompensation prediction

The best model we got for this task was trained for 36 chunks (that's less than one epoch; it overfits before reaching one epoch because there are many training samples for the same patient with different lengths).
       
       python -um mimic3models.decompensation.main --network mimic3models/keras_models/split_48.py  --dim 128 --timestep 1.0 --depth 1 --mode train --batch_size 8 --output_dir mimic3models/decompensation

### Length of stay prediction

The best model we got for this task was trained for 19 chunks.
       
       python -um mimic3models.length_of_stay.main --network mimic3models/keras_models/split_48.py  --dim 64 --timestep 1.0 --depth 1 --dropout 0.3 --mode train --batch_size 8 --partition custom --output_dir mimic3models/length_of_stay

### Phenotype classification
       
       python -um mimic3models.phenotyping.main --network mimic3models/keras_models/split_48.py --dim 256 --timestep 1.0 --depth 1 --dropout 0.3 --mode train --batch_size 8 --output_dir mimic3models/phenotyping



