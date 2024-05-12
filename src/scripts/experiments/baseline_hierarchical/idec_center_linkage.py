import torch
import torch.nn.functional as F
import torch.utils.data
from sklearn.cluster.k_means_ import k_means
from ect.methods.stacked_ae import stacked_ae
from ect.utils.logging_helper import *
from ect.utils.evaluation.dendrogram_purity import *
from scripts.Config import *
import time
from scripts.Datasets import *
from pathlib import Path
from ect.methods.DEC import DEC
from ect.utils.evaluation import cluster_acc
from sklearn.metrics import normalized_mutual_info_score as nmi
from ect.utils.dec_utils import dendrogram_purity_tree_from_clusters
import random
from ect.utils.deterministic import set_random_seed


# Data
# Import data configuration based on the parameters
from scripts.experiments.dataset_configs import *



result_dir = Path(result_main_dir, os.path.basename(__file__)[:-3], dataset_name)
result_dir.mkdir(parents=True, exist_ok=True)
config_root_logger_file_handler(f'{result_dir}/results_{time.strftime("%Y%m%d%H%M%S", time.localtime())}.log')
config_root_logger_stout()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



pt_data = torch.from_numpy(data)
pt_init_sample = torch.from_numpy(data[split_idx, :])
del data

ae_directory = Path(ae_dir, dataset_name)
check_if_aes_exist(dataset_name)

# We sort all pre-trained aes and use the same random seed for reproducibility
random.seed(42)
aes_files = sorted([x for x in ae_directory.iterdir() if x.name.endswith('.model')], key=lambda x: x.name)



def run_experiment(ae_model_path):
    logger.info(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    logger.info(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    logger.info(f"Working now on {ae_model_path.name}")
    logger.info(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    logger.info(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    new_seed = random.randint(0, 1000)
    logger.info(f"Seed value for this is: {new_seed}")
    set_random_seed(new_seed)
    train = torch.utils.data.TensorDataset(pt_data)
    train_loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True)



    n_features = pt_data.shape[1]
    # Same loss as in the DEC implementation
    ae_reconstruction_loss_fn = lambda x, y: torch.mean((x - y) ** 2)
    ae_module = stacked_ae(n_features, [500, 500, 2000, 10],
                           weight_initalizer=torch.nn.init.xavier_normal_,
                           activation_fn=lambda x: F.relu(x),
                           loss_fn=None,
                           optimizer_fn=None)

    model_data = torch.load(ae_model_path, map_location='cpu')
    ae_module.load_state_dict(model_data)
    ae_module = ae_module.cuda()

    node_data = None
    for batch_data in torch.utils.data.DataLoader(pt_init_sample, batch_size=256, shuffle=True):
        embedded_batch_np = ae_module.forward(batch_data.cuda())[0].detach().cpu().numpy()
        if node_data is None:
            node_data = embedded_batch_np
        else:
            node_data = np.concatenate([node_data, embedded_batch_np], 0)
    init_centers = k_means(node_data, n_clusters, n_init=20)[0]

    # Initialize cluster centers based on a smaller sample
    cluster_module = DEC(init_centers).cuda()
    optimizer = torch.optim.Adam(list(ae_module.parameters()) + list(cluster_module.parameters()), lr=0.001)

    def evaluate(train_round_idx, ae_module, cluster_module):
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pt_data), batch_size=256)

        pred_labels = np.zeros(pt_data.shape[0], dtype=np.int)
        index = 0
        n_batches = 0
        for batch_data in test_loader:
            batch_data = batch_data[0].cuda()
            n_batches += 1
            batch_size = batch_data.shape[0]
            embedded_data, reconstructed_data = ae_module.forward(batch_data)
            labels = cluster_module.prediction_hard_np(embedded_data)
            pred_labels[index: index + batch_size] = labels
            index = index + batch_size
        pred_tree = dendrogram_purity_tree_from_clusters(cluster_module, pred_labels, 'single')
        pred_tree2 = dendrogram_purity_tree_from_clusters(cluster_module, pred_labels, 'complete')
        lp = leaf_purity(pred_tree, gold_labels)
        leaf_purity_value = f"{lp[0]:1.3}\t({lp[1]:1.3})"
        dp_value_single = dendrogram_purity(pred_tree, gold_labels)
        dp_value_complete = dendrogram_purity(pred_tree2, gold_labels)
        logger.info(
            f"{train_round_idx} Evaluation:  leaf_purity: {leaf_purity_value}, purity_single: {dp_value_single}, purity_complete: {dp_value_complete}")
        return dp_value_single, dp_value_complete, leaf_purity_value

    evaluate("init", ae_module, cluster_module)

    n_rounds = 40000
    train_round_idx = 0
    while True:  # each iteration is equal to an epoch
        for batch_data in train_loader:
            train_round_idx += 1
            if train_round_idx > n_rounds:
                break
            batch_data = batch_data[0].cuda()

            embedded_data, reconstruced_data = ae_module.forward(batch_data)
            ae_loss = ae_reconstruction_loss_fn(batch_data, reconstruced_data)

            cluster_loss = cluster_module.loss_dec_compression(embedded_data)
            loss = cluster_loss + 0.1 * ae_loss
            if train_round_idx == 1 or train_round_idx % 100 == 0:
                logger.info(
                    f"{train_round_idx} - loss in this batch: cluster_loss:{cluster_loss.item()} "
                    f"ae_loss:{ae_loss.item()} total_loss: {ae_loss.item() + cluster_loss.item()}")

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if train_round_idx % 2000 == 0:
                evaluate(train_round_idx, ae_module, cluster_module)
        else:  # For else is being executed if break did not occur, we continue the while true loop otherwise we break it too
            continue
        break  # Break while loop here

    # Write last evaluation

    dp_value_single, dp_value_complete, leaf_purity_value = evaluate("", ae_module, cluster_module)
    result_file = Path(result_dir, f"results_{dataset_name}.txt")
    result_file_exists = result_file.exists()
    f = open(result_file, "a+")
    if not result_file_exists:
        f.write("#\"ae_model_name\"\t\"Dendrogram_Purity Single\"\t\"Dendrogram_Purity Complete\"\t\"Leaf_Purity\t(Std)\"\n")
    f.write(f"{ae_model_path.name}\t{dp_value_single}\t{dp_value_complete}\t{leaf_purity_value}\n")
    f.close()


for ae_model_path in aes_files:
    run_experiment(ae_model_path)
