'''Utitilieifor running this experiment
'''

import random
from pathlib import Path

import numpy as np

from ignite.contrib.handlers import wandb_logger
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, EarlyStopping, global_step_from_engine
from ignite.handlers import Checkpoint, DiskSaver

from ignite.utils import setup_logger

import torch
from torch.utils.data import Dataset

from audiomentations import Compose, AddGaussianNoise


def vectorized_y_true(humanresults, tag_set):
    y_true = {tag: np.zeros(len(humanresults)) for tag in tag_set}
    for i, tags in enumerate(humanresults.values()):
        # we  only look for tags in tag_set
        for tag in tag_set:
            if tag in tags:
                y_true[tag][i] = 1
            else:
                y_true[tag][i] = 0
    return y_true


def activated_output_transform(output):
    y_pred, y = output
    y_pred = torch.exp(y_pred)
    return y_pred, y


def run(model,
        dataloaders,
        optimizer,
        criterion,
        metrics,
        device,
        config,
        wandb_project_name,
        run_name=None,
        checkpoints_dir=None,
        wandb_logger_ins=None,
        taxo_names=None):

    del run_name

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']

    trainer = create_supervised_trainer(model,
                                        optimizer,
                                        criterion,
                                        device=device)
    trainer.logger = setup_logger('Trainer')

    train_evaluator = create_supervised_evaluator(model,
                                                  metrics=metrics,
                                                  device=device)
    train_evaluator.logger = setup_logger('Train Evaluator')
    validation_evaluator = create_supervised_evaluator(model,
                                                       metrics=metrics,
                                                       device=device)
    validation_evaluator.logger = setup_logger('Val Evaluator')

    test_evaluator = create_supervised_evaluator(model,
                                                 metrics=metrics,
                                                 device=device)
    test_evaluator.logger = setup_logger('Test Evaluator')

    # best_ROC_AUC -> [mean,min]
    best_ROC_AUC = [0, 0]
    best_epoch = [0, 0]

    @trainer.on(Events.EPOCH_COMPLETED, best_ROC_AUC, best_epoch, taxo_names)
    def compute_metrics(engine, best_ROC_AUC, best_epoch, taxo_names=None):
        train_evaluator.run(train_loader)
        validation_evaluator.run(val_loader)
        test_evaluator.run(test_loader)

        roc_auc_array_val = validation_evaluator.state.metrics['ROC_AUC']
        roc_auc_array_train = train_evaluator.state.metrics['ROC_AUC']
        roc_auc_array_test = test_evaluator.state.metrics['ROC_AUC']

        print('train loss', train_evaluator.state.metrics['loss'])
        print('val loss', validation_evaluator.state.metrics['loss'])
        print('test loss', test_evaluator.state.metrics['loss'])

        print('train roc auc', roc_auc_array_train, engine.state.epoch)
        print('validation roc auc', roc_auc_array_val, engine.state.epoch)
        print('test roc auc', roc_auc_array_test, engine.state.epoch)

        current_train_roc_auc_mean = np.mean(roc_auc_array_train)
        current_train_roc_auc_mean = current_train_roc_auc_mean.item()

        current_val_roc_auc_mean = np.mean(roc_auc_array_val)
        current_val_roc_auc_mean = current_val_roc_auc_mean.item()

        current_val_roc_auc_min = np.min(roc_auc_array_val)
        current_val_roc_auc_min = current_val_roc_auc_min.item()

        if current_val_roc_auc_mean > best_ROC_AUC[0]:
            best_ROC_AUC[0] = current_val_roc_auc_mean
            best_epoch[0] = engine.state.epoch
            wandb_logger_ins.log(
                {
                    'best_mean_ROC_AUC': best_ROC_AUC[0],
                    'best_mean_Epoch': best_epoch[0]
                },
                step=trainer.state.iteration)

        if current_val_roc_auc_min > best_ROC_AUC[1]:
            best_ROC_AUC[1] = current_val_roc_auc_min
            best_epoch[1] = engine.state.epoch
            wandb_logger_ins.log(
                {
                    'best_min_ROC_AUC': best_ROC_AUC[1],
                    'best_min_Epoch': best_epoch[1]
                },
                step=trainer.state.iteration)
        #log epochs seperatly to use in X axis
        wandb_logger_ins.log({'epoch': engine.state.epoch},
                             step=trainer.state.iteration)
        wandb_logger_ins.log({'val_roc_auc_mean': current_val_roc_auc_mean},
                             step=trainer.state.iteration)
        wandb_logger_ins.log({'train_roc_auc_mean': current_train_roc_auc_mean},
                             step=trainer.state.iteration)

        if taxo_names is None:
            taxo_names = [f'class_{k}' for k in range(len(roc_auc_array_val))]

        for i, taxo_name in enumerate(taxo_names):  # type: ignore
            wandb_logger_ins.log(
                {f'val_roc_auc_{taxo_name}': roc_auc_array_val[i]},
                step=trainer.state.iteration)
            wandb_logger_ins.log(
                {f'test_roc_auc_{taxo_name}': roc_auc_array_test[i]},
                step=trainer.state.iteration)

    if wandb_logger_ins is None:
        wandb_logger_ins = wandb_logger.WandBLogger(
            project=wandb_project_name,
            # name=run_name,
            config=config,
        )

    wandb_logger_ins.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,  # could add (every=100),
        tag='training',  # type: ignore
        output_transform=lambda loss: {'batchloss': loss}  # type: ignore
    )

    for tag, evaluator in [('training', train_evaluator),
                           ('validation', validation_evaluator)]:
        # Attach the logger to the evaluator on the training dataset and log NLL, Accuracy metrics after each epoch
        # We setup `global_step_transform=lambda *_: trainer.state.iteration` to take iteration value
        # of the `trainer`:
        wandb_logger_ins.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names=['loss', 'ROC_AUC'],  # type: ignore
            global_step_transform=lambda *_: trainer.state.
            iteration,  # type: ignore
        )

    wandb_logger_ins.attach_opt_params_handler(
        trainer, event_name=Events.EPOCH_COMPLETED, optimizer=optimizer)
    wandb_logger_ins.watch(model, log='all')

    def score_function_mean(engine):
        return np.mean(engine.state.metrics['ROC_AUC']).item()

    def score_function2(engine):
        print('loss', engine.state.metrics['loss'])
        return engine.state.metrics['loss']

    def score_funtion_min(engine):
        return np.min(engine.state.metrics['ROC_AUC']).item()

    if checkpoints_dir is None:
        checkpoint_dir = Path(wandb_logger_ins.run.dir) / 'checkpoints'
        checkpoints_dir.mkdir(exist_ok=True)
    else:
        wandb_logger_ins_run_dir = Path(wandb_logger_ins.run.dir)
        #run-20210429_035224-12tsplqm/files
        run_timestamp_id = wandb_logger_ins_run_dir.parent.stem.split('-')
        (wandb_logger_ins_run_id) = str(wandb_logger_ins.run.id)
        if wandb_logger_ins_run_id != run_timestamp_id[-1]:
            raise Exception(
                f' ID from wandb_logger_ins.run.dir name is not correct {wandb_logger_ins_run_id},{run_timestamp_id}'
            )
        timestamp = run_timestamp_id[1]
        checkpoint_dir = checkpoints_dir / '-'.join(
            (('run', timestamp, wandb_logger_ins_run_id)))

    # Setup object to checkpoint
    objects_to_checkpoint = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        # "lr_scheduler": lr_scheduler # we do not have scheduler
    }
    training_checkpoint = Checkpoint(
        to_save=objects_to_checkpoint,
        save_handler=DiskSaver(
            checkpoint_dir,  # type: ignore
            require_empty=False),
        n_saved=2,  # only keep last 2
        global_step_transform=lambda *_: trainer.state.epoch,
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=config['checkpoint_every_Nth_epoch']
                              ),  # how frequently save the model
        training_checkpoint)

    model_checkpoint = ModelCheckpoint(
        checkpoint_dir,  # type: ignore
        n_saved=2,
        filename_prefix='best',
        score_function=score_function_mean,
        score_name='mean_ROC_AUC',
        create_dir=True,
        # to take the epoch of the `trainer`L
        global_step_transform=global_step_from_engine(trainer),
    )

    validation_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint,
                                           {'model': model})

    if config['patience'] > 1:
        es_handler = EarlyStopping(patience=config['patience'],
                                   score_function=score_function2,
                                   trainer=trainer)
        validation_evaluator.add_event_handler(Events.COMPLETED, es_handler)

    #https://github.com/pytorch/ignite/blob/0bb3c6c0ac718258aeb0912744f9b8f3d32b7223/examples/mnist/mnist_save_resume_engine.py
    if wandb_logger_ins.run.resumed:
        # restore the best model
        checkpoint_file2resume = config['checkpointfile_2resume']
        print(f"Resume from the checkpoint: {checkpoint_file2resume} !!!! ")

        checkpoint = torch.load(checkpoint_file2resume)
        Checkpoint.load_objects(to_load=objects_to_checkpoint,
                                checkpoint=checkpoint)

#     kick everything off
    trainer.run(train_loader, max_epochs=config['epochs'])

    wandb_logger_ins.close()


# def clipped_mel_loop(XArrays, maxMelLen):
#     '''
#   maxMelLen: will clip arrays after that threshold, 850 for randomAdd
#   because it does not change original size of 10second audio files

#   ! samping rate is hard coded
#   '''
#     results = []
#     for X_array_i in range(len(XArrays)):
#         X_array = XArrays[X_array_i]
#         for index, y in enumerate(X_array):

#             mel = librosa.feature.melspectrogram(y=y.reshape(-1), sr=44100)
#             an_x = librosa.power_to_db(mel, ref=np.max)
#             an_x = an_x.astype('float32')
#             if index == 0:
#                 XMel = np.empty((X_array.shape[0], 128, maxMelLen),
#                                 dtype=np.float32)
#             XMel[index, :, :] = an_x[:, :maxMelLen]
#             # if index%100==0:
#             #     print(index)
#     #     X_array = XMel[:]
#         results.append(XMel)
#     #     print(X_array.shape)
#     return results


class audioDataset(Dataset):

    def __init__(self,
                 X,
                 y=None,
                 transform=None,
                 data_by_reference=False,
                 non_associative_labels=None):
        '''
    Args:

    '''
        self.X = X
        self.y = y
        #         self.landmarks_frame = pd.read_csv(csv_file)
        #         self.root_dir = root_dir
        self.transform = transform
        self.data_by_reference = data_by_reference
        if non_associative_labels is None:
            self.non_associative_labels = []
        else:
            self.non_associative_labels = non_associative_labels

    def __len__(self):
        if isinstance(self.X, np.ndarray):
            return self.X.shape[0]
        else:
            return len(self.X)

    def get_x(self, idx):
        if self.data_by_reference:
            x, _ = self.X[idx].get_data_by_value()
            x = torch.from_numpy(x).float()
        else:
            x = self.X[idx]
        return x

    def __getitem__(self, idx):
        x = self.get_x(idx)

        if self.y is None:
            sample = x, torch.zeros((2))
        else:
            sample = x, self.y[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


class AugmentingAudioDataset(Dataset):

    def __init__(self,
                 X,
                 y=None,
                 transform=None,
                 batch_transforms=None,
                 sampling_rate=None,
                 mix_channels_coeff=None,
                 gauss_max_amplitude=0.015,
                 data_by_reference=False,
                 non_associative_labels=None):
        '''
    Args:

    '''
        self.X = X
        self.y = y
        self.data_by_reference = data_by_reference
        #         self.landmarks_frame = pd.read_csv(csv_file)
        #         self.root_dir = root_dir
        self.transform = transform
        self.id_mix = []
        self.batch_transforms = batch_transforms
        self.sampling_rate = sampling_rate
        self.mix_channels_coeff = mix_channels_coeff
        self.gauss_max_amplitude = gauss_max_amplitude
        if self.batch_transforms is None:
            self.batch_transforms = []

        if non_associative_labels is None:
            self.non_associative_labels = []
        else:
            self.non_associative_labels = non_associative_labels
        # count y values
        #
        print(self.batch_transforms)
        if self.batch_transforms != []:
            self.row_index_4_col = self.where_is_classes_in_y(self.y)

        if 'AddGaussianNoise' in self.batch_transforms:
            self.audiomentations_augment = Compose([
                AddGaussianNoise(min_amplitude=0.01,
                                 max_amplitude=self.gauss_max_amplitude,
                                 p=0.5),
            ])

    def __len__(self):
        if isinstance(self.X, np.ndarray):
            return self.X.shape[0]
        else:
            return len(self.X)

    def get_x(self, idx):
        if self.data_by_reference:
            x, sr = self.X[idx].get_data_by_value()
            if sr != self.sampling_rate:
                print('sampling_rate not match!', sr, self.sampling_rate)
            x = torch.from_numpy(x).float()
        else:
            x = self.X[idx]
        return x

    def __getitem__(self, idx):
        sample = None

        x = self.get_x(idx)
        # Merge augmentation
        # sample = self.random_merge()
        # sample = self.random_mergev2()

        # SpecAugmentation(time_drop_width=64, time_stripes_num=2,
        # freq_drop_width=8, freq_stripes_num=2)

        # no augmentation
        if not self.batch_transforms:
            if self.y is None:
                sample = x, torch.zeros((2))
            else:
                sample = x, self.y[idx]

        if 'random_mergev2' in self.batch_transforms:  # type: ignore
            sample = self.random_mergev2(
                non_associative_labels=self.non_associative_labels)
        elif 'random_merge' in self.batch_transforms:  # type: ignore
            sample = self.random_merge(
                non_associative_labels=self.non_associative_labels)
        elif 'random_merge_fair' in self.batch_transforms:  # type: ignore
            sample = self.random_merge_fair(
                non_associative_labels=self.non_associative_labels)
        # for bath_transfrom in self.batch_transforms:

        if 'mix_channels' in self.batch_transforms:  # type: ignore
            sample = self.mix_channels(
                sample, mix_channels_coeff=self.mix_channels_coeff)

        if 'random_merge_fair' in self.batch_transforms:  # type: ignore
            if 'random_merge' in self.batch_transforms:  # type: ignore
                raise (Exception(
                    'random_merge and random_merge_fair cannot be applied together'
                ))

        if 'AddGaussianNoise' in self.batch_transforms:  # type: ignore
            x = self.audiomentations_augment(
                samples=sample[0],  # type: ignore
                sample_rate=self.sampling_rate)
            sample = (x, sample[1])  # type: ignore

        if sample is None:
            raise TypeError(
                f"augment_type is not implemented: {self.batch_transforms}")

        if self.transform:
            sample = self.transform(sample)

        return sample

    def mix_channels(self, sample, mix_channels_coeff):
        '''
            Mix audio channels by given coeff.
        '''
        #         print(sample[0].shape)
        if random.randint(0, 1) == 0:
            sample = sample[0, :] * (
                1 - mix_channels_coeff) + sample[1, :] * mix_channels_coeff
        else:
            sample = sample[1, :] * (
                1 - mix_channels_coeff) + sample[0, :] * mix_channels_coeff
        return sample

    def flip_coin_for_silence(self, y, silence_index):
        '''half of the time pick silence over everything else'''
        if y[silence_index] == 1:
            #half the time make it a silence
            if random.randint(0, 1) == 0:
                y = torch.zeros_like(y)
                y[silence_index] = 1
            #other half of the time make it a sound
            else:
                y[silence_index] = 0

        return y

    def merge_samples(self, id_1, id_2, non_associative_labels=None):
        left_y = self.y[id_1]  # type: ignore
        right_y = self.y[id_2]  # type: ignore

        left = self.get_x(id_1)
        right = self.get_x(id_2)

        x = (left + right) / 2

        y_merged = left_y + right_y
        # if only one of them is 1 then y is one anyway
        # so we just change 2s to 1
        y_merged[y_merged == 2.0] = 1
        silence_index = non_associative_labels[0]
        y_merged = self.flip_coin_for_silence(y_merged, silence_index)
        # if silence is picked, then we need to use only silent sample
        # if both are silent, then we need to use both
        if y_merged[silence_index] == 1:
            if left_y[silence_index] == 1 and right_y[silence_index] == 1:
                x = x
            else:
                x = left if left_y[silence_index] == 1 else right

        return x, y_merged

    def random_merge(self, non_associative_labels=None):
        '''
            Randomly pick two samples and merge them, with replacement.
        '''

        random_id_1 = -1
        random_id_2 = -1
        while random_id_1 == random_id_2:
            random_id_1 = random.randint(0, self.__len__() - 1)
            random_id_2 = random.randint(0, self.__len__() - 1)

        x, y = self.merge_samples(random_id_1,
                                  random_id_2,
                                  non_associative_labels=non_associative_labels)
        return x, y

    def shuffle_indexes(self,):
        self.id_mix = list(range(self.__len__()))
        random.shuffle(self.id_mix)

    def random_mergev2(self, non_associative_labels=None):
        '''
            Randomly pick two samples and merge them, without replacement.
        '''
        if len(self.id_mix) < 2:
            self.shuffle_indexes()

        random_id_1 = self.id_mix.pop()
        random_id_2 = self.id_mix.pop()

        x, y = self.merge_samples(random_id_1,
                                  random_id_2,
                                  non_associative_labels=non_associative_labels)
        return x, y

    def random_merge_fair(self, non_associative_labels=None):

        random_id_1 = self.pick_fair_sample(self.y, self.row_index_4_col)
        random_id_2 = self.pick_fair_sample(self.y, self.row_index_4_col)
        x, y = self.merge_samples(random_id_1,
                                  random_id_2,
                                  non_associative_labels=non_associative_labels)
        return x, y

    def where_is_classes_in_y(self, y):
        row_index_4_col = []
        for col_i in range(y.shape[1]):
            bb = np.argwhere(y[:, col_i] == 1)
            row_index_4_col.append(bb.flatten())
        return row_index_4_col

    def pick_fair_sample(self, y, row_index_4_col):

        number_of_classes = y.shape[1]
        # pick a random class
        random_class = (np.random.randint(0, number_of_classes))
        # pick a random sample from that class
        sample_indexes_4_class = row_index_4_col[random_class]
        fair_random_sample_id = np.random.choice(sample_indexes_4_class,
                                                 size=(1))

        return fair_random_sample_id[0]
