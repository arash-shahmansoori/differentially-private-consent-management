import torch


from utils_e2e import save_model_ckpt_cls, num_correct


class AgentSupervisedE2E:
    def __init__(self, args, device, hparams):
        self.args = args
        self.device = device
        self.hparams = hparams

        self.valid_every = args.valid_every

    def train(
        self,
        model,
        optimizer,
        ce_loss,
        input_buffer,
        epoch,
        filename_dir,
    ):
        # Set up model for training
        model.train()

        x_buffer = input_buffer["feat"]
        t_buffer = input_buffer["label"]

        optimizer.zero_grad()

        _, feat_out, _, _ = model(x_buffer)

        loss = ce_loss(feat_out, t_buffer)
        loss.backward()

        optimizer.step()

        # # Save the checkpoint for "model"
        # if epoch % self.args.save_every == 0:
        #     model.to("cpu")

        #     save_model_ckpt_cls(
        #         epoch,
        #         -1,
        #         model,
        #         None,
        #         optimizer,
        #         ce_loss,
        #         loss,
        #         None,
        #         filename_dir,
        #     )

        #     model.to(self.device)

    def accuracy_loss(
        self,
        model,
        ce_loss,
        batch_x,
        batch_y,
    ):
        total_num_correct_ = 0

        eval_model = model
        eval_model.eval()

        with torch.no_grad():
            prob, out, _, _ = eval_model(batch_x)

            corr_spk, _, _ = num_correct(prob, batch_y.view(-1), topk=1)
            total_num_correct_ += corr_spk
            acc = (total_num_correct_ / len(batch_y)) * 100

            loss = ce_loss(out, batch_y)

        return acc, loss
