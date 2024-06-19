import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human")  # 若改用這個，會畫圖
# env = gym.make("CartPole-v1", render_mode="rgb_array")
observation, info = env.reset(seed=42)
position, velocity, angle, angle_velocity = observation
score = 0
for _ in range(1000):
    env.render()
    # action = env.action_space.sample()  # 把這裡改成你的公式，看看能撐多久
    action = 1 if velocity * angle > 0 else 0
    observation, reward, terminated, truncated, info = env.step(action)
    # print('observation=', observation)
    position, velocity, angle, angle_velocity = observation
    score += reward
    if terminated or truncated:  # 這裡要加入程式，紀錄你每次撐多久
        observation, info = env.reset()
        print('done, score=', score)
        score = 0
env.close()
