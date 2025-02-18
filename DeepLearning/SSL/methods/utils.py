from torch import randn, acos, tensor, randint, cat, zeros, arange, int8, float32
from torch.nn.functional import normalize
from torch.cuda import is_available
#import matplotlib.pyplot as plt
from numpy import full
from numpy.random import choice

device = "cuda" if is_available() else "cpu"


def geodesic_distance(x, y):
    dot_product = x @ y
    
    distance = acos(dot_product)
    
    return distance

# def generate_templates(num_class, dim):
#     # each dimension of template 
#     values = tensor([-1, 0, 1])
    
#     templates = values[randint(0, len(values), (num_class, dim))].to(device)
    
#     return templates

def generate_templates(num_class, dim):
    templates = zeros((num_class, dim), dtype=float32).to(device)
    indices = randint(low=0, high=dim, size=(num_class,))
    templates[arange(num_class), indices] = randint(0, 2, (num_class,), dtype=float32).to(device) * 2 - 1
#    templates[arange(16), arange(16)] = 1
#    templates[16 + arange(16), arange(16)] = -1
    return templates

def perturbation(template, n, eps):
    dim = template.size(0)
    
    perturbation_dir = normalize(randn((n, dim)), dim=1).to(device)
    
    return normalize(template + eps * perturbation_dir)

def apply_perturbation(templates, n, eps):
    num_class = templates.size(0)
    
    base = n // num_class
    remainder = n % num_class
    
    parts = full(shape=num_class, fill_value=base)
    
    for i in choice(a=num_class, size=remainder, replace=True):
        parts[i] += 1
    
    perturbed_vs = []
    for i in range(num_class):
        perturbed_vs.append(perturbation(templates[i], parts[i], eps))
    return cat(perturbed_vs, dim=0)
    
# if __name__ == "__main__":
    # temp = generate_templates(10, 64)
    # print(temp)
#     templates = generate_templates(3, 2)
#     n = 128
#     eps = 0.1
#     perturbed_vs = apply_perturbation(templates, n, eps)
    
#     plt.figure()
#     circle = plt.Circle((0, 0), 1, fill=False, edgecolor='red', linewidth=2)
#     plt.gca().add_artist(circle)
#     plt.scatter(perturbed_vs[:,0], perturbed_vs[:, 1], alpha=0.3, cmap='viridis')
#     plt.scatter(templates[:, 0], templates[:, 1], marker="*")
#     plt.xlim(-2, 2)
#     plt.ylim(-2, 2)
#     plt.show()
