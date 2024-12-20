import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def piecewise_function(x):
    return torch.where(x < 0, x, -2 + 2 * x)

def piecewise_gradient(x):
    return torch.where(x < 0, torch.ones_like(x), 2 * torch.ones_like(x))

class SmoothModel(nn.Module):
    def __init__(self, activation_fn):
        super(SmoothModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 64),
            activation_fn(),
            nn.Linear(64, 1),
        )
        
    def forward(self, x):
        return self.fc(x)

def train_model(
    activation_fn=nn.SiLU, 
    l2_weight=0.001, 
    lr=0.001, 
    epochs=100000,
    plot_results=True,
    value_weight=1.0,
    gradient_weight=1.0
):
    # Generate training data
    x_train = torch.linspace(-1, 1, 100)
    x_train = x_train[(x_train < -0.1) | (x_train > 0.1)]
    x_train = x_train.unsqueeze(1)
    y_train = piecewise_function(x_train)
    dy_train = piecewise_gradient(x_train)

    # Model, loss, and optimizer
    model = SmoothModel(activation_fn)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_weight)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute predictions with gradient tracking
        x_train.requires_grad_(True)
        y_pred = model(x_train)
        
        # Compute gradients
        dy_pred = torch.autograd.grad(
            outputs=y_pred, 
            inputs=x_train, 
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute losses
        value_loss = mse_loss(y_pred, y_train)
        gradient_loss = mse_loss(dy_pred, dy_train)
        loss = value_weight * value_loss + gradient_weight * gradient_loss
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)
        
        # Optional: Print loss periodically
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    # Plot results
    if plot_results:

        x_test = torch.linspace(-1.5, 1.5, 200).unsqueeze(1)
        y_test = piecewise_function(x_test)
        dy_test = piecewise_gradient(x_test)
        
        x_test.requires_grad_()
        y_pred = model(x_test)
        
        # Compute gradient without creating graph
        dy_pred = torch.autograd.grad(
            outputs=y_pred, 
            inputs=x_test, 
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
            retain_graph=True
        )[0]
        
        with torch.no_grad():
            plt.figure(figsize=(10, 6))
            plt.subplot(2, 1, 1)
            plt.plot(x_test.numpy(), y_test.numpy(), label="True Value")
            plt.plot(x_test.numpy(), y_pred.numpy(), label="Predicted Value")
            
            plt.legend()
            plt.title("Value Fitting")
            
            plt.subplot(2, 1, 2)
            plt.plot(x_test.numpy(), dy_test.numpy(), label="True Gradient")
            plt.plot(x_test.numpy(), dy_pred.numpy(), label="Predicted Gradient")
            plt.legend()
            plt.title("Gradient Fitting")
            
            plt.tight_layout()
            plt.savefig('fitting.png')
            plt.show()

# Example usage
train_model(activation_fn=nn.ReLU, l2_weight=0.001,lr=0.001, epochs=10000, value_weight=1.0, gradient_weight=3.0)
