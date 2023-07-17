import controllers.rl.Agents.sac.run as sac

if __name__ == "__main__":
    sac.run(eval=True, render=True, load=True)
