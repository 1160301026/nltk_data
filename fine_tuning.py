#!/usr/bin/env python
# coding=utf-8

import torch
import sys
import collections

from transformers import BertConfig, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from ext_model import BertPair
from tqdm import tqdm
from format_converse import generate_predata
from processor import Sentihood_QA_M_Processor, convert_examples_to_features
from logger import Logger
from sklearn import metrics
from apex import amp

sys.stdout = Logger('./log/QAM_inc.log')

GPU_NUM = torch.cuda.device_count()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WEIGHT_DECAY = 0 
ADAM_EPS = 1.0e-8 # default: 1.0e-6
GRADIENT_ACCUMULATION_STEPS = 1 
LEARNING_RATE = 3e-5
BATCH_SIZE = 8
EPOCHS = 1
WARMUP_PROPORTION = 0.1

def train(model, optimizer, trainloader, testloader, pgd=None, save_model_path=None, use_apex=False):

    output_dir = './results/sentihood/QA_M/'
    print('output_dir=',output_dir)
    with open(os.path.join(output_dir, 'log.txt'), 'w') as writer:
        writer.write("epoch\tglobal_step\tloss\ttest_loss\ttest_accuracy\n")

    global_step = 0


    for epoch in trange(EPOCHS, desc='Epoch'):
        running_loss = 0.0
        running_steps = 0
        pbar = tqdm(trainloader, desc='Iteration')
        for step, data in enumerate(pbar):
            model.train()
            tokens_tensors, masks_tensors, segments_tensors, labels_ids = [t.to(DEVICE) for t in data]
            outputs = model(input_ids=tokens_tensors,
                            attention_mask=masks_tensors,
                            token_type_ids=segments_tensors,
                            labels=labels_ids)
            loss = outputs[0]
            # 注意：如果是多GPU并行，此处获得的loss就不是标量，需要取平均
            if GPU_NUM > 1:
                loss = loss.mean()

            # 接下来会有梯度累积GRADIENT_ACCUMULATION_STEPS
            
            # 设置进度条
            pbar.set_description('[epoch {}]global_step:{}, loss: {:.4f}'.format(epoch + 1, global_step, loss))

            if GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            # print(f'[step {step}] loss: {loss}')
            if use_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # 对抗训练
            # 
            
            running_loss += loss.item()
            running_steps += 1
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                model.zero_grad()
            global_step += 1

        # eval
        test_loss, test_accuracy = eval(model, testloader, output_dir=output_dir, load_tqdm=True)

        result = collections.OrderedDict()

        result = {'epoch': epoch,
                'global_step': global_step,
                'loss': running_loss/running_steps,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy}

        logger.info("***** Eval results *****")
        with open(os.path.join(output_dir, 'log.txt'), 'a+') as writer:
            for key in result.keys():
                print("  %s = %s\n", key, str(result[key]))
                writer.write("%s\t" % (str(result[key])))
            writer.write("\n")


def eval(model, dataloader, output_dir=None, load_tqdm=False):
    model.eval()
    test_loss, test_accuracy = 0, 0
    test_steps, test_examples = 0, 0
    with open(os.path.join(output_dir, "test_ep_"+str(epoch)+".txt"),"w") as test_f:

        with torch.no_grad():
            if load_tqdm:
                pbar = tqdm(dataloader)
            else:
                pbar = dataloader
            for data in pbar:
                if next(model.parameters()).is_cuda:
                    tokens_tensors, masks_tensors, segments_tensors, labels_ids = [t.cuda() for t in data if t is not None]

                model_outputs = model(input_ids=tokens_tensors,
                                attention_mask=masks_tensors,
                                token_type_ids=segments_tensors,
                                labels=labels_ids)
                tmp_test_loss, logits = model_outputs[:2]
        
                m = torch.nn.Softmax(dim=-1)
                logits = m(logits)
                logits = logits.detach().cpu().numpy()
                labels_ids = labels_ids.to('cpu').numpy()
                outputs = np.argmax(logits, axis=1)
                for output_i in range(len(outputs)):
                    f_test.write(str(outputs[output_i]))
                    for ou in logits[output_i]:
                        f_test.write(" "+str(ou))
                    f_test.write("\n")
                tmp_test_accuracy=np.sum(outputs == label_ids)

                test_loss += tmp_test_loss.mean().item()
                test_accuracy += tmp_test_accuracy

                test_examples += input_ids.size(0)
                test_steps += 1

    test_loss = test_loss / test_steps
    test_accuracy = test_accuracy / test_examples

    return test_loss, test_accuracy

def eval_score():
    pass
        # 评估 写入test_ep_x.txt里
        # aspect_Macro_F1 = metrics.f1_score(y_true, y_pred, average='macro')
        # test_f.write("aspect_Macro_F1: ", aspect_Macro_F1)



def main():
    pretrained_model_path = './data/Bert-pretrained-model/bert-base-uncased/bert-base-uncased-pytorch_model.bin'
    
    save_model_path = './data/bert_sentihood_QAM.pt'
    data_dir = './data/sentihood/bert-pair/'
    label_list = ['None', 'Positive', 'Negative']
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    processor = Sentihood_QA_M_Processor()
    train_examples = processor.get_train_examples(data_dir)
    train_features = convert_examples_to_features(train_examples, label_list, 512, tokenizer)

    test_examples = processor.get_test_examples(data_dir)
    test_features = convert_examples_to_features(test_examples, label_list, 512, tokenizer)

    train_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    train_attention_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
    train_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    train_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    test_attention_mask = torch.tensor([f.attention_mask for f in test_features], dtype=torch.long)
    test_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    test_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)


    config = BertConfig.from_pretrained('bert-base-uncased')
    print('GPU_NUM:', GPU_NUM)
    # 定义模型
    model = BertPair.from_pretrained(pretrained_model_path, config=config, freeze=False)
    print('device:', DEVICE)
    model = model.to(DEVICE)
    # pgd = PGD(model)
    print("model is ready.........")
    # 生成用于训练的dataloader
    trainset = TensorDataset(train_input_ids, train_attention_mask, train_segment_ids, train_label_ids)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    # 生成用于测试的dataloader
    testset = TensorDataset(test_input_ids, test_attention_mask, test_segment_ids, test_label_ids)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)

    params = list(model.named_parameters())
    # 下列参数，不进行正则化（权重衰减）
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                  lr=LEARNING_RATE,
                                  eps=ADAM_EPS,
                                  warmup=WARMUP_PROPORTION,
                                  weight_decay=WEIGHT_DECAY)
    if use_apex:
        print('use Apex..')
        model, optimizer = amp.initialize(model, optimizer, opt_level=apex_level)
    if GPU_NUM > 1:
        model = torch.nn.DataParallel(model)
    model.zero_grad()

    train(model, optimizer, trainloader, testloader, save_model_path=save_model_path, use_apex=True)


if __name__ = '__main__':
    main()