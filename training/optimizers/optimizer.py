import logging
import tensorflow as tf 
from training.optimizers.schedule import CustomDecay
__all__ = [ 
    'Optimizer',
]

class Optimizer():
    """
    统一管理 Optimizer
    """
    def __init__(self,learning_rate,args=None,name=None):
        self._name = name
        self._optimizer_name = args['optimizer_name'].lower()
        self._description = f"{name}_{self._optimizer_name}"
        self._mixed_precision = args['mixed_precision']
        self._learning_rate = self._schedule_check(learning_rate,args=args)
        if self._optimizer_name == 'adam':
            adam_beta_1 = float(args['adam_beta_1'])
            adam_beta_2 = float(args['adam_beta_2'])
            self._optimizer =  tf.keras.optimizers.Adam(learning_rate=self._learning_rate,beta_1=adam_beta_1,beta_2=adam_beta_2,name=self._name)
        elif self._optimizer_name == 'rmsprop':
            rmsprop_rho = float(args['rmsprop_rho'])
            rmsprop_momentum = float(args['rmsprop_momentum'])
            self._optimizer = tf.keras.optimizers.RMSprop(learning_rate=self._learning_rate,rho=rmsprop_rho,momentum=rmsprop_momentum,name=self._name)
        elif self._optimizer_name == 'sgd':
            sgd_momentum = float(args['sgd_momentum'])
            self._optimizer  = tf.keras.optimizers.SGD(learning_rate=self._learning_rate,momentum=sgd_momentum,name=self._name)
        # self = self.optimier
    def _schedule_check(self,learning_rate,args=None):
        learning_rate_schedule = args['learning_rate_schedule'].lower()
        if learning_rate_schedule == 'cosine_decay':
            decay_steps = args['cosine_decay_steps']
            alpha = args['cosine_alpha']
            lr_shcedule = tf.keras.optimizers.schedules.CosineDecay(
                            initial_learning_rate=learning_rate, 
                            decay_steps=decay_steps, 
                            alpha=alpha,name=self._name)
            logging.getLogger(__name__).info(f"""
            {self._description}
            Decay used:{learning_rate_schedule}
            initial_learning_rate:{learning_rate}
            decay_steps:{decay_steps}
            alpha:{alpha}
            """)
        elif learning_rate_schedule == 'cosine_decay_restarts':
            first_decay_steps = args['cosine_decay_restarts_first_decay_steps']
            t_mul = args['cosine_decay_restarts_t_mul']
            m_mul = args['cosine_decay_restarts_m_mul']
            alpha = args['cosine_decay_restarts_alpha']
            lr_shcedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                            initial_learning_rate=learning_rate, 
                            first_decay_steps=first_decay_steps,
                            t_mul=t_mul, 
                            m_mul=m_mul,
                            alpha=alpha,name=self._name)
            logging.getLogger(__name__).info(f"""
            {self._description} 
            Decay used:{learning_rate_schedule}
            initial_learning_rate:{learning_rate}
            first_decay_steps:{first_decay_steps}
            t_mul:{t_mul}
            m_mul:{m_mul}
            alpha:{alpha}
            """)
        elif learning_rate_schedule == 'exponential_decay':
            decay_steps = args['exponential_decay_steps']
            decay_rate = args['exponential_decay_rate']
            staircase = args['exponential_decay_staircase']
            lr_shcedule = tf.keras.optimizers.schedules.ExponentialDecay(
                            initial_learning_rate=learning_rate, 
                            decay_steps=decay_steps,
                            decay_rate=decay_rate, 
                            staircase=staircase,name=self._name)
            logging.getLogger(__name__).info(f"""
            {self._description} 
            Decay used:{learning_rate_schedule}
            initial_learning_rate:{learning_rate}
            decay_steps:{decay_steps}
            decay_rate:{decay_rate}
            staircase:{staircase}
            """)
        elif learning_rate_schedule == 'inverse_time_decay':
            decay_steps = args['inverse_time_decay_steps']
            decay_rate = args['inverse_time_decay_rate']
            staircase = args['inverse_time_decay_staircase']
            lr_shcedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                            initial_learning_rate=learning_rate, 
                            decay_steps=decay_steps,
                            decay_rate=decay_rate, 
                            staircase=staircase,name=self._name)
            logging.getLogger(__name__).info(f"""
            {self._description} 
            Decay used:{learning_rate_schedule}
            initial_learning_rate:{learning_rate}
            decay_steps:{decay_steps}
            decay_rate:{decay_rate}
            staircase:{staircase}
            """)
        elif learning_rate_schedule == 'piecewise_constant_decay':
            boundaries = args['piecewise_constant_decay_boundaries']
            values = args['piecewise_constant_decay_values']
            lr_shcedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                            boundaries=boundaries,
                            values=values,name=self._name)
            logging.getLogger(__name__).info(f"""
            f{self._description} 
            Decay used:{learning_rate_schedule}
            boundaries:{boundaries}
            values:{values}
            """)
        elif learning_rate_schedule == 'poly_nomial_decay':
            decay_steps = args['poly_nomial_decay_steps']
            end_learning_rate = args['poly_nomial_decay_end_learning_rate']
            power = args['poly_nomial_decay_power']
            cycle = args['poly_nomial_decay_cycle']
            lr_shcedule = tf.keras.optimizers.schedules.PolynomialDecay(
                            initial_learning_rate=learning_rate, 
                            decay_steps=decay_steps,
                            end_learning_rate=end_learning_rate, 
                            power=power,
                            cycle=cycle,name=self._name)
            logging.getLogger(__name__).info(f"""
            {self._description} 
            Decay used:{learning_rate_schedule}
            initial_learning_rate:{learning_rate}
            decay_steps:{decay_steps}
            end_learning_rate:{end_learning_rate}
            power:{power}
            cycle:{cycle}
            """)
        elif learning_rate_schedule == 'custom_decay':
            decay_steps = args['custom_decay_steps']
            decay_rate = args['custom_decay_rate']
            start_step = args['custom_decay_start_step']
            staircase = args['custom_decay_staircase']
            lr_shcedule = CustomDecay(
                            initial_learning_rate=learning_rate, 
                            decay_steps=decay_steps,
                            decay_rate=decay_rate, 
                            start_step = start_step,
                            staircase=staircase,name=self._name)
            logging.getLogger(__name__).info(f"""
            {self._description} 
            Decay used:{learning_rate_schedule}
            initial_learning_rate:{learning_rate}
            decay_steps:{decay_steps}
            decay_rate:{decay_rate}
            start_step:{start_step}
            staircase:{staircase}
            """)
        else:
            logging.getLogger(__name__).info(f"""
            {self._description} 
            Unsupported learning_rate_schedule:{learning_rate_schedule}
            No schedule will be used.
            """)
            lr_shcedule = learning_rate
        return lr_shcedule
    @property
    def optimier(self):
        if self._mixed_precision:
            return tf.keras.mixed_precision.LossScaleOptimizer(self._optimizer)
        else:
            return self._optimizer
# def optimizer_select(learning_rate,args):
#     optimizer_name = args['optimizer_name']
#     if optimizer_name.lower() == 'adam':
#         adam_beta_1 = float(args['adam_beta_1'])
#         adam_beta_2 = float(args['adam_beta_2'])
#         return Adam(learning_rate=learning_rate,beta_1=adam_beta_1,beta_2=adam_beta_2)
#     elif optimizer_name.lower() == 'rmsprop':
#         rmsprop_rho = float(args['rmsprop_rho'])
#         rmsprop_momentum = float(args['rmsprop_momentum'])
#         return RMSprop(learning_rate=learning_rate,rho=rmsprop_rho,momentum=rmsprop_momentum)
#     elif optimizer_name.lower() == 'sgd':
#         sgd_momentum = float(args['sgd_momentum'])
#         return SGD(learning_rate=learning_rate,momentum=sgd_momentum)


# class Adam(tf.keras.optimizers.Adam):
#     def __init__(self,*args,**kwargs):
#         super(Adam,self).__init__(*args,**kwargs)
#     def get_mixed_precision(self):
#         optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self)
#         return optimizer

# class RMSprop(tf.keras.optimizers.RMSprop):
#     def __init__(self,*args,**kwargs):
#         super(RMSprop,self).__init__(*args,**kwargs)
#     def get_mixed_precision(self):
#         optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self)
#         return optimizer

# class SGD(tf.keras.optimizers.SGD):
#     def __init__(self,*args,**kwargs):
#         super(SGD,self).__init__(*args,**kwargs)
#     def get_mixed_precision(self):
#         optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self)
#         return optimizer
if __name__ == '__main__':
    # optimizer1 = tf.keras.optimizers.Adam(2e-4)
    # optimizer1 = tf.keras.mixed_precision.LossScaleOptimizer(optimizer1)
    # print(optimizer1)
    
    optimizer1 = Adam(2e-4)
    print(optimizer1)
    optimizer1 = optimizer1.get_mixed_precision()
    print(optimizer1)
    optimizer2 = Adam(2e-4)
    print(optimizer2)
    optimizer2 = optimizer2.get_mixed_precision()
    print(optimizer2)
    optimizer3 = Adam(2e-4)
    print(optimizer3)
    optimizer3 = optimizer3.get_mixed_precision()
    print(optimizer3)