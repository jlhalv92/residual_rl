
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock2(nn.Module):
    def __init__(self, n_features, n_input, n_output,  **kwargs):
        super(ResidualBlock2, self).__init__()

        self.layer_1 = nn.Linear(n_input + 1, n_features)
        self.layer_2 = nn.Linear(n_features, n_features)
        self.layer_3 = nn.Linear(n_features, n_features)
        self.layer_out = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self.layer_1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.layer_2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.layer_3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.layer_out.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, value, state, action):
        state_action_value = torch.cat((state.float(), action.float(), value.float()), dim=1)
        Q = value
        features1 = F.relu(self.layer_1(state_action_value))
        features2 = F.relu(self.layer_2(features1))
        features3 = F.relu(self.layer_3(features2))
        rho1 = self.layer_out(features3)
        Q += rho1

        return F.relu(Q)



class QRES(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(QRES, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._in = nn.Linear(n_input, n_features)
        self._h1 = nn.Linear(n_features, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, n_output)
        self._rho_0 = ResidualBlock2(n_features, n_input, n_output)

        # nn.init.xavier_uniform_(self._in.weight, gain=nn.init.calculate_gain("relu"))
        # nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        # nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        # nn.init.xavier_uniform_(self._out.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)

        features1 = F.relu(self._in(state_action))
        features2 = F.relu(self._h1(features1))
        features3 = F.relu(self._h2(features2))
        q_0 = self._out(features3)
        q = self._rho_0(q_0, state, action)

        return torch.squeeze(q)



class ResidualBlock(nn.Module):
    def __init__(self, n_features, **kwargs):
        super(ResidualBlock, self).__init__()

        self.layer_1 = nn.Linear(n_features, n_features)
        self.layer_2 = nn.Linear(n_features, n_features)
        # self.layer_3 = nn.Linear(n_features, n_features)

        nn.init.xavier_uniform_(self.layer_1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.layer_2.weight,
                                gain=nn.init.calculate_gain('linear'))
        # nn.init.xavier_uniform_(self.layer_3.weight,
        #                         gain=nn.init.calculate_gain('linear'))

    def forward(self, x):

        identity = x
        features1 = F.relu(self.layer_1(x))
        features2 = self.layer_2(features1)
        # features3 = self.layer_3(features2)
        features2 += identity

        return F.relu(features2)


class Q2(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Q2, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._in = nn.Linear(n_input, n_features)
        self._h1 = nn.Linear(n_features, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, 3*n_features)
        self._rho_0 = ResidualBlock(3*n_features)
        self._rho_1 = ResidualBlock(3*n_features)
        self._out = nn.Linear(3*n_features, n_output)

        nn.init.xavier_uniform_(self._in.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._out.weight, gain=nn.init.calculate_gain("linear"))


    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)

        features1 = F.relu(self._in(state_action))
        features2 = F.relu(self._h1(features1))
        features3 = F.relu(self._h2(features2))
        features4 = F.relu(self._h3(features3))
        features5 = self._rho_0(features4)
        features6 = self._rho_1(features5)

        q = self._out(features6)

        return torch.squeeze(q)


class Q1(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Q1, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._in = nn.Linear(n_input, n_features)
        self._h1 = nn.Linear(n_features, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._rho_0 = ResidualBlock(n_features)
        self._out = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._in.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._out.weight, gain=nn.init.calculate_gain("linear"))


    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)

        features1 = F.relu(self._in(state_action))
        features2 = F.relu(self._h1(features1))
        features3 = F.relu(self._h2(features2))
        features4 = self._rho_0(features3)

        q = self._out(features4)

        return torch.squeeze(q)



class Q1full(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Q1full, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._in = nn.Linear(n_input, n_features)
        self._h1 = nn.Linear(n_features, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_features)
        self._h4 = nn.Linear(n_features, n_features)
        self._h5 = nn.Linear(n_features, n_features)

        # self._rho_0 = ResidualBlock(n_features)
        self._out = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._in.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h4.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._out.weight, gain=nn.init.calculate_gain("linear"))


    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)

        features1 = F.relu(self._in(state_action))
        features2 = F.relu(self._h1(features1))
        features3 = F.relu(self._h2(features2))
        features4 = F.relu(self._h3(features3))
        features5 = F.relu(self._h4(features4))
        q = self._out(features5)

        return torch.squeeze(q)
# class Q(nn.Module):
#     def __init__(self, input_shape, output_shape, n_features, **kwargs):
#         super().__init__()
#
#         n_input = input_shape[-1]
#         n_output = output_shape[0]
#
#         self._h1 = nn.Linear(n_input, n_features)
#         self._h2 = nn.Linear(n_features, n_features)
#         self._h3 = nn.Linear(n_features, n_output)
#         self._h4 = ResidualBlock(n_features, n_input, n_output)
#
#
#         nn.init.xavier_uniform_(self._h1.weight,
#                                 gain=nn.init.calculate_gain('relu'))
#         nn.init.xavier_uniform_(self._h2.weight,
#                                 gain=nn.init.calculate_gain('relu'))
#         nn.init.xavier_uniform_(self._h3.weight,
#                                 gain=nn.init.calculate_gain('linear'))
#
#     def forward(self, state, action):
#         state_action = torch.cat((state.float(), action.float()), dim=1)
#         features1 = F.relu(self._h1(state_action))
#         features2 = F.relu(self._h2(features1))
#         q_0 = self._h3(features2)
#         q_1 = self._h4(q_0, state, action)
#
#         return torch.squeeze(q_1)

class ActorNetwork(nn.Module):
    """
    Generic actor network architecture
    """
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._in = nn.Linear(n_input, n_features)
        self._h1 = nn.Linear(n_features, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._in.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._out.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, state):
        in_features = torch.squeeze(state, 1).float()

        features1 = F.relu(self._in(in_features))
        features2 = F.relu(self._h1(features1))
        features3 = F.relu(self._h2(features2))

        actions = self._out(features3)

        return actions
class CriticNetwork(nn.Module):
    """
    Generic critic network architecture
    """
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._in = nn.Linear(n_input, n_features)
        self._h1 = nn.Linear(n_features, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._in.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._out.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)

        features1 = F.relu(self._in(state_action))
        features2 = F.relu(self._h1(features1))
        features3 = F.relu(self._h2(features2))

        q = self._out(features3)

        return torch.squeeze(q)
