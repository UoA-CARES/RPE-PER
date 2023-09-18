from pydantic import BaseModel

class LearningConfig(BaseModel):
    
    seed: int
    batch_size: int
    G: int 
    max_steps_exploration: int
    max_steps_training: int
    actor_lr: float
    critic_lr: float
    gamma: float
    tau: float