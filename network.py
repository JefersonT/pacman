# Comando para treinamento:
# 	$ python3 train.py --n_workers <no. of workers> <environment name>
# Exemplo:
# 	$ python3 train.py --n_workers 8 MsPacman-v0
#
# Comando para teste:
# 	$ python3 run_checkpoint.py <environment name> <checkpoint directory>
# Exemplo:
# 	$ python3 run_checkpoint.py MsPacman-v0 runs/test-run_1552647332_cca2c03/checkpoints/

# Comando para acessar a unidade C:/ e demais pastas do Windos apartir do Ubunto WSL:
# 	$ cd /mnt/c

# Comando para visualizar o progresso do treinamento nos arquivos .log presentes dentro da pasta runs/
# 	$ tensorboard --logdir runs/
import tensorflow as tf

from multi_scope_train_op import make_train_op
from utils_tensorflow import make_grad_histograms, make_histograms, make_rmsprop_histograms, \
    logit_entropy, make_copy_ops


# Definindo função para fazer rede de inferência
def make_inference_network(obs_shape, n_actions, debug=False):
    # 'Seção 5.1. Atari Games 'no jornal diz:
    #
    #   "Treinamos tanto um agente feedforward com a mesma arquitetura quanto (Mnih et al., 2015;
    #   Nair et al., 2015; Van Hasselt et al., 2015)"
    #
    # Mnih et al. 2015 é 'Controle de nível humano através de aprendizagem de reforço profundo',
    # onde a seção de Métodos diz:
    #
    #   "A entrada para a rede neural consiste em uma imagem 84x84x4 produzida pelo
    #   mapa de pré-processamento w. A primeira camada oculta convolve 32 filtros de
    #   8x8 com passo 4 com a imagem de entrada e aplica uma retinididade não linear.
    #   A segunda camada oculta convolve 64 filtros de 4x4 com passo 2, seguido novamente por
    #   uma não linearidade do retificador, seguida por uma terceira camada convolucional
    #   que envolve 64 filtros de 3x3 com passo 1 seguido por um retificador.
    #   A camada oculta final é totalmente conectada e consiste de 512 unidades retificadoras.
    #   Camada é uma camada linear totalmente conectada com uma única saída para cada ação válida."

    # https://www.tensorflow.org/api_docs/python/tf/placeholder
    # Insere um marcador de posição(placeholder) para um tensor que será sempre alimentado.
    observations = tf.placeholder(tf.float32, [None] + list(obs_shape))

    # Numerical arguments are filters, kernel_size, strides (Argumentos numéricos são filtros, kernel_size, strides)
    # https://www.tensorflow.org/api_docs/python/tf/layers/conv2d
    # Interface funcional para a camada de convolução 2D. (descontinuada)
    # Aviso: ESTA FUNÇÃO É DEPRECADA. Ele será removido em uma versão futura. Instruções para atualização: Use keras.layers.conv2d.
    # Essa camada cria um kernel de convolução que é convolvido (na verdade, correlacionado entre si) com a entrada da camada para produzir um tensor de saídas. 
    # conv1 = tf.layers.conv2d(observations, 32, 8, 4, activation=tf.nn.relu, name='conv1')
    conv1 = tf.keras.layers.Conv2D(32, 8, 4, activation=tf.nn.relu, name='conv1')(observations)

    if debug:
        # Dump observations as fed into the network to stderr for viewing with show_observations.py. (Dump observações como alimentado na rede para stderr para visualização com show_observations.py.)
        # Imprime uma lista de tensores. (descontinuada). https://www.tensorflow.org/api_docs/python/tf/Print
        # Aviso: ESTA FUNÇÃO É DEPRECADA. Será removido após 2018-08-20. Instruções para atualização: Use tf.print em vez de tf.Print. Observe que tf.print retorna um operador sem saída que imprime diretamente a saída. Fora dos modos defuns ou ansiosos, este operador não será executado a menos que seja diretamente especificado em session.run ou usado como uma dependência de controle para outros operadores. Esta é apenas uma preocupação no modo gráfico.
        # conv1 = tf.Print(conv1, [observations], message='\ndebug observations:', summarize=2147483647)  # max no. of values to display; max int32 (no max. de valores para exibir; max int32)
        conv1 = tf.print(conv1, [observations], message='\ndebug observations:', summarize=2147483647)

    # criando segunda camada convolucional
    conv2 = tf.keras.layers.Conv2D(64, 4, 2, activation=tf.nn.relu, name='conv2')(conv1)

    # criando terceira camada convolucional

    conv3 = tf.keras.layers.Conv2D(64, 3, 1, activation=tf.nn.relu, name='conv3')(conv2)

    #print conv2.shape

    # Processando dados de conv3 para separando os valores da quantidade de filtros(f), valor de altura(h) e largura(w).
    # w, h, f = conv3.get_shape()[1:]

    # Processando saida de conv3 para Remodelando o tensor com mesmos valores do tensor conv3 e com shape [-1, int(w * h * f)], ou seja um conv1d de w*h*f posições
    # conv3_unwrapped = tf.reshape(conv3, [-1, int(w * h * f)])

    flatten = tf.keras.layers.Flatten()(conv3)
    print(flatten.shape)
    n_units = flatten.shape[1]
    flatten = tf.reshape(flatten, [-1, 1, n_units])
    features = tf.keras.layers.LSTM(256, activation='tanh', name="features")(flatten)

    # Cração de outra camada densa alimentado com pela camada anterior (features), dimenção de espaço de saido n_actions
    # Camada de saida, com os valores de saida da rede
    # action_logits = tf.layers.dense(features, n_actions, activation=None, name='action_logits')
    action_logits = tf.keras.layers.Dense(n_actions, activation=None, name='action_logits')(features)

    # Calcula as ativações softmax
    # Aviso: Alguns argumentos são preteridos: (dim). Eles serão removidos em uma versão futura. Instruções para atualização: dim está obsoleto, use o eixo
    # Esta função realiza o equivalente a
    # softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
    # transforma os dados de action_logits em dados estastisticos variando dentro de [0, 1]
    action_probs = tf.nn.softmax(action_logits)

    # Camada que mosta o valor para o dado status,
    #values = tf.layers.dense(features, 1, activation=None, name='value')
    values = tf.keras.layers.Dense(1, activation=None, name='value')(features)
    # Shape is currently (?, 1)
    # Convert to just (?)
    values = values[:, 0]

    layers = [conv1, conv2, conv3, flatten, features]

    return observations, action_logits, action_probs, values, layers


def make_loss_ops(action_logits, values, entropy_bonus, value_loss_coef, debug):
    actions = tf.placeholder(tf.int64, [None])
    returns = tf.placeholder(tf.float32, [None])

    # For the policy loss, we want to calculate log π(action_t | state_t).
    # That means we want log(action_prob_0 | state_t) if action_t = 0,
    #                    log(action_prob_1 | state_t) if action_t = 1, etc.
    # It turns out that's exactly what a cross-entropy loss gives us!
    # The cross-entropy of a distribution p wrt a distribution q is:
    #   - sum over x: p(x) * log2(q(x))
    # Note that for a categorical distribution, considering the cross-entropy of the ground truth
    # distribution wrt the distribution of predicted class probabilities, p(x) is 1 if the ground
    # truth label is x and 0 otherwise. We therefore have:
    #   - log2(q(0)) if ground truth label = 0,
    #   - log2(q(1)) if ground truth label = 1, etc.
    # So here, by taking the cross-entropy of the distribution of action 'labels' wrt the produced
    # action probabilities, we can get exactly what we want :)
    _neglogprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action_logits,#deveria ter rank=2
                                                                 labels=actions)#deveria ter rank=1
    with tf.control_dependencies([tf.assert_rank(_neglogprob, 1)]):
        neglogprob = _neglogprob

    _advantage = returns - values
    with tf.control_dependencies([tf.assert_rank(_advantage, 1)]):
        advantage = _advantage

    if debug:
        neglogprob = tf.Print(neglogprob, [actions], message='\ndebug actions:',
                              summarize=2147483647)
        advantage = tf.Print(advantage, [returns], message='\ndebug returns:',
                             summarize=2147483647)

    policy_entropy = tf.reduce_mean(logit_entropy(action_logits))

    # Note that the advantage is treated as a constant for the policy network update step.
    # We're calculating advantages on-the-fly using the value approximator. This might make us
    # worry: what if we're using the loss for training, and the advantages are calculated /after/
    # training has changed the network? But for A3C, we don't need to worry, because we compute the
    # gradients separately from applying them.
    #
    # Note also that we want to maximise entropy, which is the same as minimising negative entropy.
    policy_loss = neglogprob * tf.stop_gradient(advantage)
    policy_loss = tf.reduce_mean(policy_loss) - entropy_bonus * policy_entropy
    value_loss = value_loss_coef * tf.reduce_mean(0.5 * advantage ** 2)
    loss = policy_loss + value_loss

    return actions, returns, advantage, policy_entropy, policy_loss, value_loss, loss

### Aqui
class Network:

    def __init__(self, scope, n_actions, entropy_bonus, value_loss_coef, max_grad_norm, optimizer,
                 add_summaries, detailed_logs=False, debug=False):

        with tf.variable_scope(scope):

            observations, action_logits, action_probs, value, layers = \
                make_inference_network(obs_shape=(84, 84, 4), n_actions=n_actions, debug=debug)

            actions, returns, advantage, policy_entropy, policy_loss, value_loss, loss = \
                make_loss_ops(action_logits, value, entropy_bonus, value_loss_coef, debug)

        sync_with_global_op = make_copy_ops(from_scope='global', to_scope=scope)

        train_op, grads_norm = make_train_op(loss, optimizer,
                                             compute_scope=scope, apply_scope='global',
                                             max_grad_norm=max_grad_norm)

        self.states = observations
        self.action_probs = action_probs
        self.value = value
        self.actions = actions
        self.returns = returns
        self.advantage = advantage
        self.policy_entropy = policy_entropy
        self.policy_loss = policy_loss
        self.value_loss = value_loss
        self.loss = loss
        self.layers = layers

        self.sync_with_global_ops = sync_with_global_op
        self.optimizer = optimizer
        self.train_op = train_op
        self.grads_norm = grads_norm

        if add_summaries:
            self.summaries_op = self.make_summary_ops(scope, detailed_logs)
        else:
            self.summaries_op = None

    def make_summary_ops(self, scope, detailed_logs):
        variables = tf.trainable_variables(scope)
        grads_policy = tf.gradients(self.policy_loss, variables)
        grads_value = tf.gradients(self.value_loss, variables)
        grads_combined = tf.gradients(self.loss, variables)
        grads_norm_policy = tf.global_norm(grads_policy)
        grads_norm_value = tf.global_norm(grads_value)
        grads_norm_combined = tf.global_norm(grads_combined)

        scalar_summaries = [
            ('rl/policy_entropy', self.policy_entropy),
            ('rl/advantage_mean', tf.reduce_mean(self.advantage)),
            ('loss/loss_policy', self.policy_loss),
            ('loss/loss_value', self.value_loss),
            ('loss/loss_combined', self.loss),
            ('loss/grads_norm_policy', grads_norm_policy),
            ('loss/grads_norm_value', grads_norm_value),
            ('loss/grads_norm_combined', grads_norm_combined),
            ('loss/grads_norm_combined_clipped', self.grads_norm),
        ]
        summaries = []
        for name, val in scalar_summaries:
            summary = tf.summary.scalar(name, val)
            summaries.append(summary)

        if detailed_logs:
            summaries.extend(make_grad_histograms(variables, grads_combined))
            summaries.extend(make_rmsprop_histograms(self.optimizer))
            summaries.extend(make_histograms(self.layers, 'activations'))
            summaries.extend(make_histograms(variables, 'weights'))

        return tf.summary.merge(summaries)
