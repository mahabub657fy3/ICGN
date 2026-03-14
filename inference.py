import argparse
from utils import *

def main():
    parser = argparse.ArgumentParser(description='ICGN')
    parser.add_argument('--test_dir', default=r'results_imagenet_res50_325k_c50/gan_c50/res50')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--model_t',type=str, default= 'normal')
    parser.add_argument('--label_flag', type=str, default='C50', help='Label nums: N8, C20, C50, ...')
    parser.add_argument('--dataset', type=str, default='imagenet',choices=['imagenet', 'cifar10'], help='Dataset for target model')
    args = parser.parse_args()
    print(args)

    # GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    dic = dict()

    if args.model_t == 'all':
        model_name_list = ['vgg16', 'googlenet', 'incv3', 'res50', 'res152', 'dense121', 'incv4', 'inc_res_v2', 'adv_incv3', 'ens_inc_res_v2']
    elif args.model_t == 'robust':
        model_name_list = ['adv_incv3', 'ens_inc_res_v2', 'res50_sin', 'res50_sin_in', 'res50_sin_fine_in']
    elif args.model_t == 'normal':
        model_name_list = [ 'incv3', 'vgg16', 'googlenet', 'res50', 'res152', 'dense121']
    elif args.model_t == 'cifar':
        #CIFAR-10 classifiers
        model_name_list = ['cifar10_vgg19_bn','cifar10_vgg16_bn','cifar10_vgg13_bn','cifar10_resnet56','cifar10_resnet44','cifar10_resnet32','cifar10_resnet20',]
    else:
        model_name_list = [args.model_t]

    for model_name in model_name_list:
        print(f'\n=== Evaluating target model: {model_name} ===')

        # Select model & normalization
        if args.dataset == 'imagenet':
            model_t = load_model(model_name)
            norm_fn = normalize_imagenet
        else:  # cifar10
            model_t = load_cifar10_model(model_name, device=device)
            norm_fn = normalize_cifar10

        model_t = model_t.to(device)
        model_t.eval()

        # Image size
        if args.dataset == 'cifar10':
            img_size = 32
        else:
            if model_name in ['incv3', 'incv4', 'inc_res_v2', 'adv_incv3', 'ens_inc_res_v2']:
                img_size = 255
            else:
                img_size = 224

        # Setup-Data
        data_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

        class_ids = get_classes(args.label_flag, dataset=args.dataset)

        # Evaluation
        sr = np.zeros(len(class_ids))
        for idx in range(len(class_ids)):
            test_dir = '{}_t{}'.format(args.test_dir, class_ids[idx])

            target_acc = 0.
            target_test_size = 0.
            test_set = datasets.ImageFolder(test_dir, data_transform)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
            for i, (img, _) in enumerate(test_loader):
                img = img.to(device)
                adv_out = model_t(norm_fn(img.clone().detach()))
                target_acc += torch.sum(adv_out.argmax(dim=-1) == (class_ids[idx])).item()
                target_test_size += img.size(0)
            sr[idx] = target_acc / target_test_size
            print('sr: {}'.format(sr))
        print('model:{} \t target acc:{:.2%}\t target_test_size:{}'.format(model_name, sr.mean(), target_test_size))
        dic[model_name] = sr.mean() * 100
    print(dic)
if __name__ == "__main__":
    main() 
