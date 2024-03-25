
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, n_features, n_input, n_output,  **kwargs):
        super(ResidualBlock, self).__init__()

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
        q_old = value
        features1 = F.relu(self.layer_1(state_action_value))
        features2 = F.relu(self.layer_2(features1))
        features3 = F.relu(self.layer_3(features2))
        rho = self.layer_out(features3)
        q = q_old + rho
        return q


class ResidualBlockFeatures(nn.Module):
    def __init__(self, n_features, n_input, n_output,  **kwargs):
        super(ResidualBlockFeatures, self).__init__()

        self.layer_1 = nn.Linear(n_input + n_features, 2*n_features)
        self.layer_2 = nn.Linear(2*n_features, 2*n_features)
        self.layer_3 = nn.Linear(2*n_features, 2*n_features)
        self.layer_out = nn.Linear(2*n_features, n_output)

        nn.init.xavier_uniform_(self.layer_1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.layer_2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.layer_3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.layer_out.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, value, state, action, features):
        state_action_features = torch.cat((state.float(), action.float(), features.float()), dim=1)
        q_old = value
        features1 = F.relu(self.layer_1(state_action_features))
        features2 = F.relu(self.layer_2(features1))
        features3 = F.relu(self.layer_3(features2))
        rho = self.layer_out(features3)
        q = rho + q_old
        return q

class QRES_Features(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(QRES_Features, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]
        self._in = nn.Linear(n_input, n_features)
        self._h1 = nn.Linear(n_features, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, n_output)

        self._rho_0 = ResidualBlockFeatures(n_features, n_input, n_output)


    def forward(self, state, action, old_q=False, rho=False):
        state_action = torch.cat((state.float(), action.float()), dim=1)

        features1 = F.relu(self._in(state_action))
        features2 = F.relu(self._h1(features1))
        features3 = F.relu(self._h2(features2))
        q_0 = self._out(features3)
        q = self._rho_0(q_0, state, action, features3)

        if old_q:
            return torch.squeeze(q_0)
        if rho:
            return torch.squeeze(q-q_0)

        return torch.squeeze(q)



class ResidualBlockFeaturesSlim(nn.Module):
    def __init__(self, n_features, n_input, n_output,  **kwargs):
        super(ResidualBlockFeaturesSlim, self).__init__()
        n_features_res = n_input + n_features

        self.layer_1 = nn.Linear(n_features_res, n_features_res)
        self.layer_2 = nn.Linear(n_features_res, n_features_res)
        self.layer_3 = nn.Linear(n_features_res, n_features_res)
        self.layer_out = nn.Linear(n_features_res, n_output)

        nn.init.xavier_uniform_(self.layer_1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.layer_2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.layer_3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.layer_out.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, value, state, action, features):
        state_action_features = torch.cat((state.float(), action.float(), features.float()), dim=1)
        q_old = value
        features1 = F.relu(self.layer_1(state_action_features))
        features2 = F.relu(self.layer_2(features1))
        features3 = F.relu(self.layer_3(features2))
        rho = self.layer_out(features3)
        q = rho + q_old
        return q

class QRES_FeaturesSlim(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(QRES_FeaturesSlim, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]
        self._in = nn.Linear(n_input, n_features)
        self._h1 = nn.Linear(n_features, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, n_output)

        self._rho_0 = ResidualBlockFeaturesSlim(n_features, n_input, n_output)


    def forward(self, state, action, old_q=False, rho=False):
        state_action = torch.cat((state.float(), action.float()), dim=1)

        features1 = F.relu(self._in(state_action))
        features2 = F.relu(self._h1(features1))
        features3 = F.relu(self._h2(features2))
        q_0 = self._out(features3)
        q = self._rho_0(q_0, state, action, features3)

        if old_q:
            return torch.squeeze(q_0).detach()
        if rho:
            return torch.squeeze(q-q_0).detach()

        return torch.squeeze(q)




class QRESLIM(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(QRESLIM, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]
        self._in = nn.Linear(n_input, n_features)
        self._h1 = nn.Linear(n_features, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, n_output)

        self._rho_0 = ResidualBlock(n_features, n_input, n_output)


    def forward(self, state, action, old_q=False, rho=False):
        state_action = torch.cat((state.float(), action.float()), dim=1)

        features1 = F.relu(self._in(state_action))
        features2 = F.relu(self._h1(features1))
        features3 = F.relu(self._h2(features2))

        q_0 = self._out(features3)

        q = self._rho_0(q_0, state, action)

        if old_q:
            return torch.squeeze(q_0).detach()

        if rho:
            return torch.squeeze(q-q_0).detach()

        return torch.squeeze(q)




class QRESLIM2(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(QRESLIM2, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._in = nn.Linear(n_input, n_features)
        self._h1 = nn.Linear(n_features, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, n_output)

        self._rho_0 = ResidualBlock(n_features, n_input, n_output)
        self._rho_1 = ResidualBlock(n_features, n_input, n_output)


    def forward(self, state, action, old_q=False, rho=False):
        state_action = torch.cat((state.float(), action.float()), dim=1)

        features1 = F.relu(self._in(state_action))
        features2 = F.relu(self._h1(features1))
        features3 = F.relu(self._h2(features2))
        q_0 = self._out(features3)
        q_1 = self._rho_0(q_0, state, action)
        q = self._rho_1(q_1, state, action)

        if old_q:
            return torch.squeeze(q_1).detach()

        if rho:
            return torch.squeeze(q-q_1).detach()

        return torch.squeeze(q)


class QRESLIM3(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(QRESLIM3, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._in = nn.Linear(n_input, n_features)
        self._h1 = nn.Linear(n_features, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, n_output)

        self._rho_0 = ResidualBlock(n_features, n_input, n_output)
        self._rho_1 = ResidualBlock(n_features, n_input, n_output)
        self._rho_2 = ResidualBlock(n_features, n_input, n_output)


    def forward(self, state, action,  old_q=False, rho=False):
        state_action = torch.cat((state.float(), action.float()), dim=1)

        features1 = F.relu(self._in(state_action))
        features2 = F.relu(self._h1(features1))
        features3 = F.relu(self._h2(features2))
        q_0 = self._out(features3)
        q_1 = self._rho_0(q_0, state, action)
        q_2 = self._rho_1(q_1, state, action)
        q = self._rho_2(q_2, state, action)

        if old_q:
            return torch.squeeze(q_2).detach()

        if rho:
            return torch.squeeze(q-q_2).detach()

        return torch.squeeze(q)


class QRESLIM4(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(QRESLIM4, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._in = nn.Linear(n_input, n_features)
        self._h1 = nn.Linear(n_features, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, n_output)

        self._rho_0 = ResidualBlock(n_features, n_input, n_output)
        self._rho_1 = ResidualBlock(n_features, n_input, n_output)
        self._rho_2 = ResidualBlock(n_features, n_input, n_output)
        self._rho_3 = ResidualBlock(n_features, n_input, n_output)


    def forward(self, state, action,  old_q=False, rho=False):
        state_action = torch.cat((state.float(), action.float()), dim=1)

        features1 = F.relu(self._in(state_action))
        features2 = F.relu(self._h1(features1))
        features3 = F.relu(self._h2(features2))
        q_0 = self._out(features3)
        q_1 = self._rho_0(q_0, state, action)
        q_2 = self._rho_1(q_1, state, action)
        q_3 = self._rho_2(q_2, state, action)
        q = self._rho_3(q_3, state, action)

        if old_q:
            return torch.squeeze(q_3).detach()

        if rho:
            return torch.squeeze(q-q_3).detach()

        return torch.squeeze(q)





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


class TransferCritic(nn.Module):
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
        self._h3 = nn.Linear(n_features, n_features)
        self._h4 = nn.Linear(n_features, n_features)
        self._h5 = nn.Linear(n_features, n_features)
        self._out_q = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h4.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h5.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._out_q.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)

        features1 = F.relu(self._in(state_action))
        features2 = F.relu(self._h1(features1))
        features3 = F.relu(self._h2(features2))
        features4 = F.relu(self._h3(features3))
        features5 = F.relu(self._h4(features4))
        features6 = F.relu(self._h5(features5))

        q = self._out_q(features6)

        return torch.squeeze(q)