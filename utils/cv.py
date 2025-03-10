

class EarlyStopping:
    def __init__(self, save_path, patience=5, verbose=False, delta=-0.01):
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.best_f1_ma = 0
        self.auc = 0
        self.delta = delta

    def __call__(self, epoch, f1_macro, auc, model):
        score = f1_macro
        if score < self.best_f1_ma + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0

        if self.best_f1_ma < f1_macro or (self.best_f1_ma == f1_macro and auc >= self.auc):
            self.save_checkpoint(epoch, f1_macro, auc, model)

    def save_checkpoint(self, epoch, f1_macro, auc, model):
        if self.verbose:
            print(f'F1_macro increased ({self.best_f1_ma:.6f} --> {f1_macro:.6f}). Saving model ...')
    
        torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'f1_macro': f1_macro
                }, self.save_path)
        self.best_f1_ma = f1_macro
        self.auc = auc

class Trainer():
    def __init__(self, model, lr, optimizer_name, max_epoch, batch_size, gradient_accumulation_steps, cuda=False):
        self.model = model
        self.cuda = cuda
        if cuda:
            self.model = self.model.to('cuda')
        self.lr = lr
        
        weight_p, bias_p = [], []
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
                
        # self.optimizer = torch.optim.Adam(
        #     [{'params': weight_p, 'weight_decay': weight_decay},
        #      {'params': bias_p, 'weight_decay': 0}
        #      ], lr=lr
        # )
        if optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        elif optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError
            
            
        self.max_epoch = max_epoch
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        # self.loss_func = BCEFocalLoss()
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def train(self, trial, outdir, trainset_reader, validset_reader, testset_reader):
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        writer = SummaryWriter(log_dir='./log')
        # scheduler = StepLR(self.optimizer, step_size=40, gamma=0.1)
        early_stopping = EarlyStopping(outdir + 'checkpoint.pt', patience=10, verbose=True)
    
        global_step = 0
        records = defaultdict(default_value)
        for epoch in range(self.max_epoch):
            # training
            self.model.train()
            # optimizer.zero_grad()
            train_loss = 0
            predictions = []
            labels = []
            for batch_idx, (inputs, targets) in enumerate(trainset_reader):
                targets = targets.float()
                if self.cuda:
                    inputs = tuple(input_tensor.to('cuda') for input_tensor in inputs)
                    targets = targets.to('cuda')
                    
                logits = self.model(*inputs)
                loss = self.loss_func(logits, targets)
                train_loss += loss.item()
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
                self.optimizer.zero_grad()
                loss.backward()
                global_step += 1
                if global_step % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                
                predictions += torch.sigmoid(logits).to('cpu').tolist()
                labels += targets.to('cpu').tolist()
            # update learning rate
            # scheduler.step()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(tag=name+'_grad', values=param.grad.clone().cpu().data.numpy(), global_step=epoch + 1)
                    writer.add_histogram(tag=name+'_data', values=param.clone().cpu().data.numpy(), global_step=epoch + 1)
                
            
            train_loss = train_loss / len(trainset_reader)
            records['loss']['train'].append(train_loss)
            train_evaluator = Evaluator(predictions, labels)
            records['auc']['train'].append(train_evaluator.auc())
            records['precision_0']['train'].append(train_evaluator.precision(pos_class=0))
            records['precision_1']['train'].append(train_evaluator.precision(pos_class=1))
            records['recall_0']['train'].append(train_evaluator.recall(pos_class=0))
            records['recall_1']['train'].append(train_evaluator.recall(pos_class=1))
            records['f1_0']['train'].append(train_evaluator.f1(pos_class=0))
            records['f1_1']['train'].append(train_evaluator.f1(pos_class=1))
            records['f1_macro']['train'].append(train_evaluator.f1_macro())
            records['f1_micro']['train'].append(train_evaluator.f1_micro())
            records['accuracy']['train'].append(train_evaluator.accuracy())
            if self.cuda:
                torch.cuda.empty_cache()

            # validating
            valid_loss = 0
            predictions = []
            labels = []
            self.model.eval()
            with torch.no_grad():
                for index, (inputs, targets) in enumerate(validset_reader):
                    targets = targets.float()
                    if self.cuda:
                        inputs = tuple(input_tensor.to('cuda') for input_tensor in inputs)
                        targets = targets.to('cuda')
        
                    logits = self.model(*inputs)
                    loss = self.loss_func(logits, targets)
                    valid_loss += loss.item()
                    
                    labels += targets.to('cpu').tolist()
                    predictions += torch.sigmoid(logits).to('cpu').tolist()
                assert len(predictions) == len(labels)
                
            valid_loss = valid_loss / len(validset_reader)
            records['loss']['valid'].append(valid_loss)
            valid_evaluator = Evaluator(predictions, labels)
            records['auc']['valid'].append(valid_evaluator.auc())
            records['precision_0']['valid'].append(valid_evaluator.precision(pos_class=0))
            records['precision_1']['valid'].append(valid_evaluator.precision(pos_class=1))
            records['recall_0']['valid'].append(valid_evaluator.recall(pos_class=0))
            records['recall_1']['valid'].append(valid_evaluator.recall(pos_class=1))
            records['f1_0']['valid'].append(valid_evaluator.f1(pos_class=0))
            records['f1_1']['valid'].append(valid_evaluator.f1(pos_class=1))
            records['f1_macro']['valid'].append(valid_evaluator.f1_macro())
            records['f1_micro']['valid'].append(valid_evaluator.f1_micro())
            records['accuracy']['valid'].append(valid_evaluator.accuracy())

            # testing
            test_loss = 0
            predictions = []
            labels = []
            self.model.eval()
            with torch.no_grad():
                for index, (inputs, targets) in enumerate(testset_reader):
                    targets = targets.float()
                    if self.cuda:
                        inputs = tuple(input_tensor.to('cuda') for input_tensor in inputs)
                        targets = targets.to('cuda')
        
                    logits = self.model(*inputs)
                    loss = self.loss_func(logits, targets)
                    test_loss += loss.item()
                    
                    labels += targets.to('cpu').tolist()
                    predictions += torch.sigmoid(logits).to('cpu').tolist()
                assert len(predictions) == len(labels)
                
            test_loss = test_loss / len(testset_reader)
            records['loss']['test'].append(test_loss)
            test_evaluator = Evaluator(predictions, labels)
            records['auc']['test'].append(test_evaluator.auc())
            records['f1_macro']['test'].append(test_evaluator.f1_macro())
            records['f1_micro']['test'].append(test_evaluator.f1_micro())
            records['accuracy']['test'].append(test_evaluator.accuracy())
            
            auc_score = records['auc']['valid'][-1]
            print(f"epoch={epoch}\ttrain_loss={train_loss:.6f}\tvalid_loss={valid_loss:.6f}\tvalid_accuray={records['accuracy']['valid'][-1]:.6f}\tF1_macro={records['f1_macro']['valid'][-1]:.6f}\tF1_micro={records['f1_micro']['valid'][-1]:.6f}\tauc={auc_score:.6f}")
            if trial != None:
                trial.report(records['accuracy']['valid'][-1], epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                
            early_stopping(epoch, records['f1_macro']['valid'][-1], records['auc']['valid'][-1], self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                # break

        writer.close()
        return records

def cross_validation(model_args, configs, k_fold=5):
    fold_num = []
    for i in range(k_fold):
        dataset_path1 = f'/kaggle/input/politifact/PolitiFact/mapped_data/5fold/train_{i}.tsv'
        dataset_path2 = f'/kaggle/input/politifact/PolitiFact/mapped_data/5fold/test_{i}_ori.tsv'
        
        trainset = pd.read_csv(dataset_path1, sep='\t', usecols=['cred_label', 'claim_text', 'claim_source', 'evidence', 'evidence_source'])
        testset = pd.read_csv(dataset_path2, sep='\t', usecols=['cred_label', 'claim_text', 'claim_source', 'evidence', 'evidence_source'])
        
        train_json_file_path = f'train_{i}.jsonl'
        test_json_file_path = f'test_{i}.jsonl'
        
        df2json(trainset, train_json_file_path)
        df2json(testset, test_json_file_path)
        
        trainset = FCDataset(train_json_file_path, tokenizer, bert_model, cuda=True)
        testset = FCDataset(test_json_file_path, tokenizer, bert_model, cuda=True)
        if len(testset.examples) % 32 == 1:
            testset.examples = testset.examples[:-1]

        train_sampler = BucketSampler(trainset, bucket_size=bucket_size, shuffle=True)
        trainset_reader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn)
        test_sampler = BucketSampler(testset, bucket_size=bucket_size, shuffle=True)
        testset_reader = DataLoader(testset, batch_size=batch_size, sampler=test_sampler, collate_fn=collate_fn)

        model = FCModel(**model_args)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    
        trainer = Trainer(model, **configs)
        records_tmp = trainer.train(None, f'model/{i}/', trainset_reader, validset_reader, testset_reader)
        plot_metric(metrics=['loss', 'accuracy'], sources=['train', 'valid', 'test'], records=records_tmp)
        plot_metric(metrics=['auc', 'f1_micro', 'f1_macro'], sources=['train', 'valid', 'test'], records=records_tmp)
        
        if i == 0:
            records = records_tmp
            fold_num += [1] * len(records['loss']['train'])
        else:
            target_length = max(len(records['loss']['train']), len(records_tmp['loss']['train']))
            fold_num += [0] * (target_length - len(fold_num))
            fold_num[: len(records_tmp['loss']['train'])] = [fold_num[i] + 1 for i in range(len(records_tmp['loss']['train']))]
            for metric in records:
                for source in ['train', 'valid', 'test']:
                    records[metric][source] = records[metric][source] + [0] * (target_length - len(records[metric][source]))
                    records_tmp[metric][source] = records_tmp[metric][source] + [0] * (target_length - len(records_tmp[metric][source]))
                    records[metric][source] = [records[metric][source][i] + records_tmp[metric][source][i] for i in range(len(records[metric][source]))]

        model.to('cpu')
        del model
        gc.collect()
    
    for metric in records:
        for source in ['train', 'valid', 'test']:
            records[metric][source] = [records[metric][source][i] / fold_num[i] for i in range(len(records[metric][source]))]
            
    return records