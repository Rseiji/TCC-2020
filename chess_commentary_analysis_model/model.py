##############################
## BERT - Model
##############################

def cross_validation(input_ids, attention_masks, labels):
  dataset = TensorDataset(input_ids, attention_masks, labels)
  train_size = int(0.9 * len(dataset))
  val_size = len(dataset) - train_size

  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

  print('{:>5,} training samples'.format(train_size))
  print('{:>5,} validation samples'.format(val_size))

  return train_dataset, val_dataset


def data_loader(batch_size, train_dataset, val_dataset):
    train_dataloader = DataLoader(
              train_dataset,
              sampler = RandomSampler(train_dataset),
              batch_size = batch_size
          )

    validation_dataloader = DataLoader(
              val_dataset,
              sampler = SequentialSampler(val_dataset),
              batch_size = batch_size
          )
    return train_dataloader, validation_dataloader


def get_model(bert_model='bert-base-uncased', num_labels=2):
    model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=bert_model,
        num_labels=num_labels,
        output_attentions = False,
        output_hidden_states = False
    )
    model.cuda()
    return model


def get_adam_optimizer(model, learning_rate=2e-5, epsilon=1e-8):
  optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )
  return optimizer


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    """Takes a time in seconds and returns a string hh:mm:ss
    """
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train_epoch(train_dataloader, optimizer, scheduler, model, total_train_loss):
    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        model.zero_grad()        
        loss, logits = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    return (optimizer, scheduler, total_train_loss)


def validate_epoch(validation_dataloader, model):
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():        
            (loss, logits) = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask,
                                labels=b_labels)
            
        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)
    return total_eval_accuracy, total_eval_loss


def classification_iterate(train_dataloader, validation_dataloader, optimizer, scheduler, model, device, training_stats, total_t0):
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_train_loss = 0
        model.train()

        optimizer, scheduler, total_train_loss = train_epoch(train_dataloader, optimizer, scheduler, model, total_train_loss)
        avg_train_loss = total_train_loss / len(train_dataloader)            
        training_time = format_time(time.time() - t0)

        print(" Average training loss: {0:.2f}".format(avg_train_loss))
        print(" Training epcoh took: {:}".format(training_time))

        print("")
        print("Running Validation...")
        t0 = time.time()
        model.eval()

        total_eval_accuracy, total_eval_loss = validate_epoch(validation_dataloader, model)
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)

        print(" Accuracy: {0:.2f}".format(avg_val_accuracy))
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        
        validation_time = format_time(time.time() - t0)
        
        print(" Validation Loss: {0:.2f}".format(avg_val_loss))
        print(" Validation took: {:}".format(validation_time))

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    return training_stats


def predict(input_ids, attention_masks, labels, device, batch_size, epochs, learning_rate=2e-5, epsilon=1e-8):
    model = get_model(bert_model='bert-base-uncased', num_labels=2)
    train_dataset, val_dataset = cross_validation(input_ids, attention_masks, labels)
    train_dataloader, validation_dataloader = data_loader(batch_size=batch_size, train_dataset=train_dataset, val_dataset=val_dataset)
    optimizer = get_adam_optimizer(model, learning_rate=learning_rate, epsilon=epsilon)

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    training_stats = []
    total_t0 = time.time()
    training_stats = classification_iterate(train_dataloader, validation_dataloader, optimizer, scheduler, model, device, training_stats, total_t0)
    return training_stats
