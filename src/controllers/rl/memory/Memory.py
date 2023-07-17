import numpy as np

"""
Replay memory buffer
"""


class Memory:
    def __init__(self, capacity, state_shape, action_shape, device):
        """
        Creates a new replay buffer with the supplied capacity.

        :param capacity: Size of the buffer.
        :param state_shape: Shape / dimensions of the state space.
        :param action_shape: Shape / dimensions of the action space.
        :param device: The device being use.
        """
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.is_image = len(state_shape) == 3
        self.state_type = np.uint8 if self.is_image else np.float32

        self.reset()

    def append(self, state, action, reward, next_state, done, episode_done=None):
        """
        Adds a new transition to the buffer.

        :param state: The start state.
        :param reward: The reward gained.
        :param next_state: The ending state.
        :param done: 1 if trajectory was terminated, 0 otherwise.
        :param episode_done: unused (it was used in a failed prioritized memory buffer)
        """
        self._append(state, action, reward, next_state, done)

    def _append(self, state, action, reward, next_state, done):
        """
        Internal method to actually add the transition to buffer,
        and update internal pointers.

        :param state: The start state.
        :param reward: The reward gained.
        :param next_state: The ending state.
        :param done: 1 if trajectory was terminated, 0 otherwise.
        """
        state = np.array(state, dtype=self.state_type)
        next_state = np.array(next_state, dtype=self.state_type)

        self.states[self._p] = state
        self.actions[self._p] = action
        self.rewards[self._p] = reward
        self.next_states[self._p] = next_state
        self.dones[self._p] = done

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

    def sample(self, batch_size):
        """
        Uniformly randomly samples transitions from the buffer.

        :param batch_size: The size of the batch to sample.
        """
        indices = np.random.randint(low=0, high=self._n, size=batch_size)
        return self._sample(indices)

    def _sample(self, indices):
        """
        Fetches the transitions at the specified indices
        and casts the results to tensors.

        :param indices: The indices to extract.
        """
        if self.is_image:
            states = self.states[indices].astype(np.uint8)
            next_states = self.next_states[indices].astype(np.uint8)
            states = torch.ByteTensor(states).to(self.device).float() / 255.0
            next_states = torch.ByteTensor(next_states).to(self.device).float() / 255.0
        else:
            states = self.states[indices]
            next_states = self.next_states[indices]
            states = torch.FloatTensor(states).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)

        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Returns the number of stored transitions.
        """
        return self._n

    def reset(self):
        """
        Resets this buffer.
        """
        self._n = 0
        self._p = 0

        self.states = np.empty(
            (self.capacity, *self.state_shape), dtype=self.state_type
        )
        self.actions = np.empty((self.capacity, *self.action_shape), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.next_states = np.empty(
            (self.capacity, *self.state_shape), dtype=self.state_type
        )
        self.dones = np.empty((self.capacity, 1), dtype=np.float32)

    def get(self):
        """
        Returns the entire buffer.
        """
        valid = slice(0, self._n)
        return (
            self.states[valid],
            self.actions[valid],
            self.rewards[valid],
            self.next_states[valid],
            self.dones[valid],
        )

    def load(self, batch):
        """
        Loads a batch of transitions into the buffer.

        :param batch: The batch to load.
        """
        num_data = len(batch[0])

        if self._p + num_data <= self.capacity:
            self._insert(slice(self._p, self._p + num_data), batch, slice(0, num_data))
        else:
            mid_index = self.capacity - self._p
            end_index = num_data - mid_index
            self._insert(slice(self._p, self.capacity), batch, slice(0, mid_index))
            self._insert(slice(0, end_index), batch, slice(mid_index, num_data))

        self._n = min(self._n + num_data, self.capacity)
        self._p = (self._p + num_data) % self.capacity

    def _insert(self, mem_indices, batch, batch_indices):
        """
        Loads the specified batch indices in the memory
        indices supplied.

        :param mem_indices: The memory indices to overwrite.
        :param batch: The batch to be written from.
        :param batch_indices: The indices to write from the batch.
        """
        states, actions, rewards, next_states, dones = batch
        self.states[mem_indices] = states[batch_indices]
        self.actions[mem_indices] = actions[batch_indices]
        self.rewards[mem_indices] = rewards[batch_indices]
        self.next_states[mem_indices] = next_states[batch_indices]
        self.dones[mem_indices] = dones[batch_indices]
