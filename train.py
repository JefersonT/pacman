#!/usr/bin/env python3

# Diversas interfaces do sistema operacional. Este módulo fornece uma maneira portátil de usar a funcionalidade dependente do sistema operacional.
import os

# Este módulo implementa algumas funções úteis em nomes de caminhos.
import os.path as osp

# Este módulo fornece várias funções relacionadas ao tempo. Para funcionalidades relacionadas, veja também os módulos datetimee calendar.
import time

# Este módulo constrói interfaces de threading de nível superior na parte superior do _threadmódulo de nível inferior .
# A classe Thread representa uma atividade que é executada em um thread separado de controle.
from threading import Thread

# Você está fazendo um protótipo de algo e quer ser capaz de representar graficamente algum valor sem passar por todas as etapas usuais para configurar corretamente o registro do TensorFlow?
# O easy-tf-log é um módulo simples para fazer isso.
import easy_tf_log

# TensorFlow é uma biblioteca de código aberto para aprendizado de máquina aplicável a uma ampla variedade de tarefas. É um sistema para criação e treinamento de redes neurais para detectar e decifrar padrões e correlações, análogo à forma como humanos aprendem e raciocinam.
import tensorflow as tf

# Arquivo utils.py.
import utils

# Arquivo utils_tensorflow.py.
import utils_tensorflow

# Arquivo env.py.
from env import make_envs

# Arquivo network.py.
from network import Network, make_inference_network

# Arquivo params.py.
from params import parse_args

# Arquivo utils_tensorflow.py.
from utils_tensorflow import make_lr, make_optimizer

# Aqruivo worker.py.
from worker import Worker

# Um objeto de mapeamento que representa o ambiente de string. Por exemplo, environ['HOME']é o nome do caminho do seu diretório pessoal (em algumas plataformas) e é equivalente getenv("HOME")em C.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # filter out INFO messages (filtrar mensagens INFO)

# Definindo função para Fazer a rede.
def make_networks(n_workers, obs_shape, n_actions, value_loss_coef, entropy_bonus, max_grad_norm,
                  optimizer, detailed_logs, debug):
    # https://www.tensorflow.org/api_docs/python/tf/Graph notes that graph construction isn't
    # thread-safe. So we all do all graph construction serially before starting the worker threads.
    # Porturgues:
    #   (https://www.tensorflow.org/api_docs/python/tf/Graph observa que a construção do grafo não é thread-safe.
    #   Portanto, todos nós fazemos toda a construção de grafos em série antes de iniciar os threads de trabalho.)

    # A with instrução é usada para envolver a execução de um bloco com métodos definidos por um gerenciador de contexto (consulte a seção Com gerenciadores de contexto de instrução ).
    # Um gerenciador de contexto para definir ops que cria variáveis ​​(camadas).
    # Esse gerenciador de contexto valida que o (opcional) valuesé do mesmo gráfico, garante que o gráfico seja o gráfico padrão e envia um escopo de nome e um escopo de variável
    # Create shared parameters (Crie parâmetros compartilhados)
    with tf.variable_scope('global'): # https://www.tensorflow.org/api_docs/python/tf/variable_scope#class_variable_scope
        make_inference_network(obs_shape, n_actions) # Função importado de network.py Cria a estrutura da rede


    # Create per-worker copies of shared parameters (Criar cópias por trabalhador dos parâmetros compartilhados)
    worker_networks = []
    for worker_n in range(n_workers): # loop n_workers vezes
        create_summary_ops = (worker_n == 0) # create_summary_ops se for o primeiro loop
        worker_name = "worker_{}".format(worker_n) # defino o nome do worker de acordo com o loop

        # Network = função de network.py
        ##### AQUI
        network = Network(scope=worker_name, n_actions=n_actions, entropy_bonus=entropy_bonus,
                          value_loss_coef=value_loss_coef, max_grad_norm=max_grad_norm,
                          optimizer=optimizer, add_summaries=create_summary_ops,
                          detailed_logs=detailed_logs, debug=debug)
        worker_networks.append(network) # inserindo network na lista

    return worker_networks # retorna lista de redes


# Definindo a fução para criação de workers
def make_workers(sess, envs, networks, n_workers, log_dir):
    print("Starting {} workers".format(n_workers))
    workers = []
    for worker_n in range(n_workers):
        worker_name = "worker_{}".format(worker_n)
        worker_log_dir = osp.join(log_dir, worker_name)
        w = Worker(sess=sess, env=envs[worker_n], network=networks[worker_n],
                   log_dir=worker_log_dir)
        workers.append(w)

    return workers


def run_worker(worker, n_steps_to_run, steps_per_update, step_counter, update_counter):
    while int(step_counter) < n_steps_to_run:
        steps_ran = worker.run_update(steps_per_update)
        step_counter.increment(steps_ran)
        update_counter.increment(1)


# definindo função que inicia a thread de cada worker
def start_worker_threads(workers, n_steps, steps_per_update, step_counter, update_counter):
    worker_threads = [] # inicia lista de threads por workes

    # Para cada worker fazer:
    for worker in workers:
        def f(): # define uma função f que roda o worker ocorrente
            run_worker(worker, n_steps, steps_per_update, step_counter, update_counter)
        thread = Thread(target=f) # TENHO DE PESQUISAR DIREITO
        thread.start() # inica a thread
        worker_threads.append(thread) # add a thread a lista de threads
    return worker_threads


def run_manager(worker_threads, sess, lr, step_counter, update_counter, log_dir, saver,
                wake_interval_seconds, ckpt_interval_seconds):
    checkpoint_file = osp.join(log_dir, 'checkpoints', 'network.ckpt') # Junte um ou mais componentes de caminho de maneira inteligente. O valor de retorno é a concatenação de caminho e qualquer membro de * caminhos com exatamente um separador de diretório ( os.sep) seguindo cada parte não vazia, exceto a última, significando que o resultado só terminará em um separador se a última parte estiver vazia. Se um componente for um caminho absoluto, todos os componentes anteriores serão descartados e a junção continuará a partir do componente de caminho absoluto.

    ckpt_timer = utils.Timer(duration_seconds=ckpt_interval_seconds)
    ckpt_timer.reset()

    step_rate = utils.RateMeasure()
    step_rate.reset(int(step_counter))

    while True:
        time.sleep(wake_interval_seconds)

        steps_per_second = step_rate.measure(int(step_counter))
        easy_tf_log.tflog('misc/steps_per_second', steps_per_second)
        easy_tf_log.tflog('misc/steps', int(step_counter))
        easy_tf_log.tflog('misc/updates', int(update_counter))
        easy_tf_log.tflog('misc/lr', sess.run(lr))

        alive = [t.is_alive() for t in worker_threads]

        if ckpt_timer.done() or not any(alive):
            saver.save(sess, checkpoint_file, int(step_counter))
            print("Checkpoint saved to '{}'".format(checkpoint_file))
            ckpt_timer.reset()

        if not any(alive):
            break


def main():
    args, lr_args, log_dir, preprocess_wrapper = parse_args() # parse_args() é importado de params
    easy_tf_log.set_dir(log_dir) # seta o caminho dos logs em easy_ty_log

    utils_tensorflow.set_random_seeds(args.seed) # iniciando a semente aleatóriamente
    sess = tf.Session() # Uma classe para executar operações do TensorFlow. Um Sessionobjeto encapsula o ambiente no qual os Operation objetos são executados e os Tensorobjetos são avaliados. 

    envs = make_envs(args.env_id, preprocess_wrapper, args.max_n_noops, args.n_workers,
                     args.seed, args.debug, log_dir)

    step_counter = utils.TensorFlowCounter(sess)
    update_counter = utils.TensorFlowCounter(sess)
    lr = make_lr(lr_args, step_counter.value)
    optimizer = make_optimizer(lr)

    # Criando o conjunto de redes por threads
    networks = make_networks(n_workers=args.n_workers, obs_shape=envs[0].observation_space.shape,
                             n_actions=envs[0].action_space.n, value_loss_coef=args.value_loss_coef,
                             entropy_bonus=args.entropy_bonus, max_grad_norm=args.max_grad_norm,
                             optimizer=optimizer, detailed_logs=args.detailed_logs,
                             debug=args.debug)

    # Retorna todas as variáveis ​​criadas com trainable=True.
    # scope: (Opcional.) Uma string. Se fornecida, a lista resultante é filtrada para incluir apenas itens cujo nameatributo corresponde ao scopeuso re.match
    global_vars = tf.trainable_variables('global')


    # Por que save_relative_paths = True?
    # De modo que o arquivo de 'checkpoint' em texto simples use caminhos relativos,
    # para que possamos restaurar a partir de pontos de verificação criados em outra máquina.
    saver = tf.train.Saver(global_vars, max_to_keep=1, save_relative_paths=True)

    # se existir um checkpoint para carregar ele restaura os dados para proceguir de onde parou, caso contrário ele inicia do 0
    if args.load_ckpt:
        print("Restoring from checkpoint '{}'...".format(args.load_ckpt), end='', flush=True)
        saver.restore(sess, args.load_ckpt) # restaura(carrega) a sessão do checkpoint especificado
        print("done!")
    else:
        sess.run(tf.global_variables_initializer())

    # Criando as workes
    workers = make_workers(sess, envs, networks, args.n_workers, log_dir)

    # inicia as threads referente a cada workers criada
    worker_threads = start_worker_threads(workers, args.n_steps, args.steps_per_update,
                                          step_counter, update_counter)

    # Gerenciador de execução das workers_threads
    run_manager(worker_threads, sess, lr, step_counter, update_counter, log_dir, saver,
                args.manager_wake_interval_seconds, args.ckpt_interval_seconds)

    for env in envs:
        env.close()


if __name__ == '__main__':
    main() # chamando a função principal
