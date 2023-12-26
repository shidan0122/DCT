import pickle

def load_dataset(name):
    train_loc = 'data/'+name+'/clip_train.pkl'
    test_loc = 'data/'+name+'/clip_test.pkl'
    with open(train_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        train_labels = data['label']
        train_texts = data['text']
        train_images = data['image']
        
    with open(test_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        test_labels = data['label']   
        test_texts = data['text']
        test_images = data['image']

    data_dict = {'train_img': train_images, 'train_txt':train_texts, 'train_label':train_labels, 'test_img':test_images, 'test_txt':test_texts, 'test_label':test_labels}
    return data_dict