from csv import DictReader
from csv import DictWriter
from torch.utils.data import Dataset
import pickle

label_ref = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}

#label_ref_rev = {0: 'agree', 1: 'disagree', 2: 'discuss', 3: 'unrelated'}
label_ref_rev = {0: 'unrelated', 1: 'agree', 2: 'disagree', 3: 'discuss'}
class FNCData:

    """
    Define class for Fake News Challenge data
    """

    def __init__(self, file_instances, file_bodies):

        # Load data
        self.instances = self.read(file_instances)
        bodies = self.read(file_bodies)
        self.heads = {}
        self.bodies = {}

        # Process instances
        for instance in self.instances:
            if instance['Headline'] not in self.heads:
                head_id = len(self.heads)
                self.heads[instance['Headline']] = head_id
            instance['Body ID'] = int(instance['Body ID'])

        # Process bodies
        for body in bodies:
            self.bodies[int(body['Body ID'])] = body['articleBody']

    
    def read(self, filename):

        """
        Read Fake News Challenge data from CSV out_file
        Args:
            filename: str, filename + extension
        Returns:
            rows: list, of dict per instance
        """

        # Initialise
        rows = []

        # Process out_file
        with open(filename, "r", encoding='utf-8') as table:
            r = DictReader(table)
            for line in r:
                rows.append(line)

        return rows

def save_predictions(data, pred_file, file):

    """
    Save predictions to CSV file
    Args:
        pred: numpy array, of numeric predictions
        file: str, filename + extension
    """
    with open(file, 'w') as csvfile, open(pred_file, 'rb') as p:
        pred = pickle.load(p)
        fieldnames = ['Headline', 'Body ID', 'Stance']
        writer = DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for origin_dict, label in zip(data.instances, pred):
            writer.writerow({ 'Headline': origin_dict['Headline'],
                              'Body ID': origin_dict['Body ID'],
                'Stance': label_ref_rev[label]})


class FNCDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data.instances)
    
    def __getitem__(self, index):
        headline = self.data.instances[index]['Headline']
        body_id = self.data.instances[index]['Body ID']
        body = self.data.bodies[body_id]

        if 'Stance' in self.data.instances[index].keys():
            stance = self.data.instances[index]['Stance']
            stance_label = label_ref[stance]
            return headline, body, stance_label

        return headline, body