import numpy as np
import sys
from scipy.optimize import line_search


def nn_opt(x0, grd, nn_idcs=None, opt_itrs=1000, step_sched=lambda i : 1./(i+1), b1=0.9, b2=0.999, eps=1e-8, verbose=False):
  x = x0.copy()
  m1 = np.zeros(x.shape[0])
  m2 = np.zeros(x.shape[0])
  for i in range(opt_itrs):
    print("Step {} out of {}".format(i,opt_itrs))
    g = grd(x)
    if verbose:
      active_idcs = np.intersect1d(nn_idcs, np.where(x==0)[0])
      inactive_idcs = np.setdiff1d(np.arange(x.shape[0]), active_idcs)
      sys.stdout.write('itr ' + str(i+1) +'/'+str(opt_itrs)+': ||inactive constraint grads|| = ' + str(np.sqrt((g[inactive_idcs]**2).sum())) + '                \r')
      sys.stdout.flush()
    m1 = b1*m1 + (1.-b1)*g
    m2 = b2*m2 + (1.-b2)*g**2
    upd = step_sched(i)*m1/(1.-b1**(i+1))/(eps + np.sqrt(m2/(1.-b2**(i+1))))
    x -= upd
    #project onto x>=0
    if nn_idcs is None:
        x = np.maximum(x, 0.)
    else:
        x[nn_idcs] = np.maximum(x[nn_idcs], 0.)

  if verbose:
    sys.stdout.write('\n')
    sys.stdout.flush()
  return x


def an_opt(x0, grd, search_direction, grad_norm_variance, opt_itrs=20):
  print("Performing Newton Optimization:")

  # Define starting point
  x = x0.copy()
  print("Newton Step: 0")
  # Estimate variance for relevant terms for tuning projection dimension
  samples_var = grad_norm_variance(x)
  print("Newton Step Gradiant Variance: {}".format(samples_var))

  # Use a line search to take a good initial step
  g = search_direction(x)
  g_u = grd(x)

  print("a of gradient is: {}".format(np.dot(g_u, g)))

  a=1.0
  k=0.9
  fail = 0
  upd = 0
  print("Optimizing step size:")
  while np.all(x+a*g < 0):
      a /= 1.5

  for j in range(10):
    test = np.dot(grd(np.maximum(x+a*g, 0.)),g)/np.dot(g_u,g)
    print("test = {}".format(test))
    if test >= k or test <= 0:
      a = a/1.5
    else:
      upd = a*g
      print("alpha = {}".format(a))
      print("a = {}".format(np.dot(grd(np.maximum(x + upd, 0.)), g)))
      fail = 1
      break

  if fail == 0:
    return x

  x += upd
  # project onto x>=0
  x = np.maximum(x, 0.)
  # Take the rest of the steps with step size 1
  norm_grd_old = 0
  for i in range(opt_itrs):
    print("Newton Step: {}".format(i+1))
    g = search_direction(x)
    g_u = grd(x)
    # Calculate relative difference between current and previous gradient estimate
    norm_grd = np.sqrt(np.sum(g_u**2))
    rel_g_u = (norm_grd_old - norm_grd)/norm_grd
    if i == 0:
        rel_g_u = 1
    norm_grd_old = norm_grd
    # Early stopping conditions:
    # Small relative difference from previous gradient
    if rel_g_u <= 0.1:
        break

    print("a of gradient is: {}".format(np.dot(g_u,g)))

    x += g
    # project onto x>=0
    x = np.maximum(x, 0.)

  return x

