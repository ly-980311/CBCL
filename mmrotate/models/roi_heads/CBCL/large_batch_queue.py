import torch
import torch.nn as nn


class Large_batch_queue(nn.Module):
    """
    Labeled matching of OIM loss function.
    """

    def __init__(self, num_classes=5532, number_of_instance=2, selection=0.7, feat_len=256):
        """
        Args:
            num_classes (int): Number of labeled classes.
            feat_len (int): Length of the feature extracted by the network.
        """
        super(Large_batch_queue, self).__init__()
        self.num_classes = num_classes
        self.number_of_instance = number_of_instance
        self.selection = selection
        self.register_buffer("large_batch_queue", torch.zeros(num_classes * number_of_instance, feat_len))
        self.register_buffer("queue_label", torch.zeros(num_classes * number_of_instance))
        self.register_buffer("tail", torch.tensor(0).long())

    def forward(self, features, pid_labels):
        """
        Args:
            features (Tensor[N, feat_len]): Features of the proposals.
            pid_labels (Tensor[N]): Ground-truth classes of the proposals.

        Returns:
            scores (Tensor[N, num_persons]): Labeled matching scores, namely the similarities
                                             between proposals and labeled objects.
        """

        with torch.no_grad():
            judge = torch.tensor(0).long()  #
            for indx, label in enumerate(torch.unique(pid_labels)):
                label = int(label)  #
                # if label >= 0:
                    # self.large_batch_queue[self.tail] = torch.mean(features[pid_labels == label], dim=0)
                    # self.queue_label[self.tail] = label
                    # self.tail += 1
                    # if self.tail >= self.large_batch_queue.shape[0]:
                    #     self.tail -= self.large_batch_queue.shape[0]
                if 0 <= label < self.num_classes:  #
                    if (torch.max(torch.matmul(self.large_batch_queue[label], features[indx])) < self.selection
                        and judge == 1) or judge == 0:  #
                        self.large_batch_queue[self.tail] = features[indx]
                        self.queue_label[self.tail] = label
                        self.tail += 1
                        if self.tail >= self.large_batch_queue.shape[0]:
                            self.tail -= self.large_batch_queue.shape[0]
                            judge = 1
        return self.large_batch_queue, self.queue_label
