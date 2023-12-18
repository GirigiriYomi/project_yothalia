import csv
import random
import torch
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser

class BenchmarkDataset(Dataset):
    def __init__(self, csv_file):
        self.data = self.load_data(csv_file)
        self.num_sample = len(self.data)

    def load_data(self, csv_file):
        data = []
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 4 and row[0]:
                    data.append({
                        'id': row[0],
                        'instruction': row[1],
                        'user': row[2],
                        'assisstant': row[3]
                    })
        return data

    def __len__(self):
        return self.num_sample

    def __getitem__(self, idx):
        sample = self.data[idx]
        instruction = sample['instruction']
        user = sample['user']
        correct_assisstant = sample['assisstant']

        # Get 4 random outputs
        potential_assistant = []
        random_idx = random.sample(range(self.num_sample), 4)
        potential_assistant = [self.data[idx]['assisstant'] for idx in random_idx]

        # Insert the correct output at a random position (Also the correct answer position)
        correct_position = random.randint(0, 4)
        potential_assistant.insert(correct_position, correct_assisstant)

        # Get prompt 
        prompt = self.get_prompt(instruction, user, potential_assistant)

        return idx, prompt, correct_position, correct_assisstant
    
    def get_prompt(self, instruction, user, potential_assistant):
        prompt = (
            f'<s>[INST]<<SYS>>\n'
            f'Please select the best response with the given instruction and input. Return with its No. number of response.<</SYS>>\n'
            f"Instruction: {instruction}.\n"
            f'Input: {user}.\n'
            f'Response:\n'
        )
        
        for i, response in enumerate(potential_assistant):
            prompt += f'No.{i}: {response}\n'
        
        prompt += '[/INST]'
        return prompt
    

class FinetuneDataset(Dataset):
    def __init__(self, csv_file):
        self.data = self.load_data(csv_file)
        self.num_sample = len(self.data)

    def load_data(self, csv_file):
        data = []
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 4 and row[0]:
                    data.append({
                        'id': row[0],
                        'instruction': row[1],
                        'user': row[2],
                        'assisstant': row[3]
                    })
        return data

    def __len__(self):
        return self.num_sample

    def __getitem__(self, idx):
        sample = self.data[idx]
        instruction = sample['instruction']
        user = sample['user']
        assisstant = sample['assisstant']

        # Get prompt 
        prompt = self.get_prompt(instruction, user, assisstant)

        return idx, prompt
    
    def get_prompt(self, instruction, user, assisstant, history = ''):
        prompt = f'<s>[INST]<<SYS>>\n'\
                f'{instruction}<</SYS>>\n'\
                f'{history}\n'\
                f'{user}[/INST]\n'\
                f'{assisstant}</s>'
        return prompt
    

class RoleplayDataloader(DataLoader):
    def __init__(self, args) -> None:
        self.args = args

    def get_ft_dataloader(self, file):
        finetune_dataset = FinetuneDataset(self.args.path + file)
        train_loader = DataLoader(finetune_dataset, batch_size=self.args.batch_size, shuffle=True)
        return train_loader
    
    def get_bm_dataloader(self, file):
        benchmark_dataset = BenchmarkDataset(self.args.path + file)
        test_dataloader = DataLoader(benchmark_dataset, batch_size=self.args.batch_size, shuffle=False)
        return test_dataloader


# Example usage
def parse():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=1, help='batch size')
    parser.add_argument('--path', default='../../train_sample/csv/', help='optimizer for training')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    random.seed(328)
    args = parse()
    role_playdataloader = RoleplayDataloader(args)

    # get ft dataloader with zh_roleplay
    zh_ft_train_loader = role_playdataloader.get_ft_dataloader('zh_roleplay.csv')
    for batch in zh_ft_train_loader:
        idx, prompt = batch
        print(idx)
        print(prompt)
        break

    # get bm dataloader with eng_roleplay
    eng_bm_test_loader = role_playdataloader.get_bm_dataloader('eng_roleplay.csv')
    for batch in eng_bm_test_loader:
        idx, prompt, correct_position, correct_assisstant = batch
        print(idx)
        print(prompt)
        print(correct_position)
        print(correct_assisstant)
        break
    

    