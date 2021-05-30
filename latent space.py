def plot_latent_space(latent_variable, label, epoch):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, 45)
    ax.grid(linestyle='dotted')    
    ax.set_xlabel(r'activation [$z_1$]', weight='normal', fontsize=12)
    ax.set_ylabel(r'activation [$z_2$]', weight='normal', fontsize=12)
    ax.set_zlabel(r'activation [$z_3$]', weight='normal', fontsize=12)
    plt.title('latent space activations at epoch ' + str(epoch), fontsize=12)
    regular = latent_variable[np.where(label == 'regular')]
    ax.scatter(regular[:, 0], regular[:, 1], regular[:, 2], c='C0', alpha=0.4, marker="o"
    anomalies_1 = latent_variable[np.where(label == 'global')]
    ax.scatter(anomalies_1[:, 0], anomalies_1[:, 1], anomalies_1[:, 2], c='C1', s=100, marker="^")
    anomalies_2 = latent_variable[np.where(label == 'local')]
    ax.scatter(anomalies_2[:, 0], anomalies_2[:, 1], anomalies_2[:, 2], c='C2', s=100, marker="^")

def get_latent_space(encoder_model_name):
    encoder_eval = encoder()
    encoder_eval.load_state_dict(torch.load(os.path.join("models", encoder_model_name)))
    data = autograd.Variable(torch_dataset)
    encoder_eval.eval()
    latent_variable = encoder_eval(data).data.numpy()
    return latent_variable

eval_epoch = 5
z_representation = get_latent_space(encoder_model_name='ep_{}_encoder_model.pth'.format(eval_epoch))
plot_latent_space(z_representation, label, epoch=eval_epoch)