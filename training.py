import argparse
import datetime
import time
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AdamW, RobertaForSequenceClassification, RobertaTokenizer, get_linear_schedule_with_warmup, pad_sequences


def train(model, train_dataloader, optimizer, scheduler, device):
    model.train()

    total_train_loss = 0

    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()

        result = model(b_input_ids,
                       token_type_ids=None,
                       attention_mask=b_input_mask,
                       labels=b_labels,
                       return_dict=True)

        loss = result.loss
        total_train_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)

    return avg_train_loss


def evaluate(model, test_dataloader, device):
    model.eval()

    total_test_loss = 0
    predictions, true_labels = [], []

    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            result = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels,
                           return_dict=True)

        loss = result.loss
        total_test_loss += loss.item()

        logits = result.logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend(logits.argmax(axis=1))
        true_labels.extend(label_ids)

    avg_test_loss = total_test_loss / len(test_dataloader)
    accuracy = accuracy_score(true_labels, predictions)

    return avg_test_loss, accuracy

def train_and_evaluate_model(sentences, labels, fold, tokenizer, device):
    # Create the dataset
    input_ids = []
    attention_masks = []
    
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(sent,
                                             add_special_tokens=True,
                                             max_length=128,
                                             pad_to_max_length=True,
                                             return_attention_mask=True,
                                             return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    
    dataset = TensorDataset(input_ids, attention_masks, labels)
    
    # Set up the data loader
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    train_indices, test_indices = list(kfold.split(dataset))[fold]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=32)
    
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)
    
    # Instantiate the model
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model.to(device)
    
    # Set up the optimizer and scheduler
    optimizer = AdamW(model.parameters(),
                      lr=5e-5,
                      eps=1e-8)
    
    total_steps = len(train_dataloader) * 5
    warmup_steps = int(total_steps * 0.1)
    
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    
    # Train and evaluate the model
    for epoch in range(5):
        start_time = time.time()
    
        avg_train_loss = train(model, train_dataloader, optimizer, scheduler, device)
    
        avg_test_loss, accuracy = evaluate(model, test_dataloader, device)
    
        end_time = time.time()
        epoch_time = str(datetime.timedelta(seconds=int(end_time - start_time)))
    
        print(f'Fold {fold} | Epoch {epoch + 1} | Train Loss: {avg_train_loss:.3f} | '
              f'Test Loss: {avg_test_loss:.3f} | Test Acc: {accuracy:.3f} | Time: {epoch_time}')
    
    return avg_train_loss, avg_test_loss, accuracy


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_dataset = MyDataset(args.train_data_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    model = MyModel().to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train model
    for epoch in range(args.num_epochs):
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss
            if i % args.print_freq == 0:
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # Save trained model
    torch.save(model.state_dict(), args.output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save trained model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--print_freq", type=int, default=100, help="Print frequency")

    args = parser.parse_args()

    main(args)