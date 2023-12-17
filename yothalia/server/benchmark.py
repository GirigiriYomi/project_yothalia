import os
import openai
import random
import re
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from dotenv import load_dotenv
from dataset import RoleplayDataloader

# Set your OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def openai_request(payload):
    """
    Sends a request to the OpenAI GPT-3.5-turbo model to generate a completion based on the provided user payload (Just for example).

    Args:
        payload (str): The user's input or prompt to be processed by the model.

    Returns:
        str: The generated completion or response from the GPT-3.5-turbo model.
    """
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": payload},
        ]
    )
    result = response.choices[0].message.content
    return result


def evaluate(args, files):
    """
    Evaluates the performance of a model on a set of roleplay files by comparing predicted choices 
    against correct choices for each prompt. The evaluation assumes a batch size of 1 (free to change).

    Args:
        args (Namespace): Command-line arguments containing model and evaluation configurations.
        files (list): List of roleplay files to be evaluated.

    Returns:
        tuple: A tuple containing two lists - the list of evaluated files and the corresponding accuracy values.
    """
    role_play_dataloader = RoleplayDataloader(args)
    accs = []
    # evaluate all files
    for file in files:
        bm_dataloader = role_play_dataloader.get_bm_dataloader(file)
        count_total = len(bm_dataloader.dataset)
        count_correct = 0
        for batch in bm_dataloader:
            idx, prompt, correct_choice, correct_assisstant = batch
            
            # TODO PLACE TO CHANGE WITH OUR MODEL
            result = openai_request(prompt[0])

            predict_choice = result_postprocessing(result)
            if predict_choice == int(correct_choice[0]):
                count_correct += 1
                
        acc = count_correct / count_total
        accs.append(acc)
    return accs


def result_postprocessing(result):
    """
    Extracts and returns the numeric value following the pattern "No.<num>" in the input string.

    Args:
        result (str): The input string.
    Returns:
        int: The numeric value following the pattern, or -1 if no match is found.
    """
    pattern = r'No.(\d+)'
    match = re.search(pattern, result)
    try:
        num = match.group(1)
        return int(num)
    except:
        return -1
    
def plot_accuracy(files, accs):
    """
    Plot the accuracy for each file.

    Args:
        files (list): List of file names.
        accs (list): List of accuracy values corresponding to each file.
    """
    plt.bar(files, accs, color='blue')
    plt.xlabel('File')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for Each File')
    plt.ylim([0, 1])  # Set the y-axis range from 0 to 1
    plt.show()


def parse():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=1, help='batch size')
    parser.add_argument('--path', default='../../train_sample/csv/', help='optimizer for training')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    random.seed(328)
    args = parse()

    # evaluate with specific csv file
    files = ['eng_roleplay.csv', 'eng_roleplay2.csv', 'zh_roleplay.csv']
    accs = evaluate(args, files)
    print(accs)

    # plot bar graph of acc
    plot_accuracy(files, accs)
