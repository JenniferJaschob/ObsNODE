import warnings
import torch
import torch.nn as nn
from scipy import interpolate

from odeint.odeint import *
from misc.miscellaneous import check_inputs_adjoint, _flat_to_shape, _mixed_norm



class OdeintAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, shapes, node_func, obs_func, t, y_obs, rtol, atol, method, options, event_fn, adjoint_rtol, adjoint_atol, adjoint_method,
                adjoint_options, t_requires_grad, node_par_len, obs_par_len, *adjoint_params):

        ctx.shapes = shapes
        ctx.node_func = node_func
        ctx.obs_func = obs_func
        ctx.adjoint_rtol = adjoint_rtol
        ctx.adjoint_atol = adjoint_atol
        ctx.adjoint_method = adjoint_method
        ctx.adjoint_options = adjoint_options
        ctx.t_requires_grad = t_requires_grad
        ctx.event_mode = event_fn is not None
        ctx.node_par_len = node_par_len
        ctx.obs_par_len = obs_par_len

        with torch.no_grad():
            z_now = obs_func(y_obs)
            ans = odeint(node_func, z_now, t, rtol=rtol, atol=atol, method=method, options=options, event_fn=event_fn)

            if event_fn is None:
                z = ans
                ctx.save_for_backward(t, z, z_now, y_obs, *adjoint_params)
            else:
                event_t, z = ans
                ctx.save_for_backward(t, z, z_now , y_obs, event_t, *adjoint_params)

        return ans

    @staticmethod
    def backward(ctx, *grad_z_l):

        with torch.no_grad():
             node_func = ctx.node_func
             obs_func = ctx.obs_func
             adjoint_rtol = ctx.adjoint_rtol
             adjoint_atol = ctx.adjoint_atol
             adjoint_method = ctx.adjoint_method
             adjoint_options = ctx.adjoint_options
             t_requires_grad = ctx.t_requires_grad
             node_par_len = ctx.node_par_len
             obs_par_len = ctx.obs_par_len


             # FROM ORIGINAL SCRIPT, HERE NOT USEABLE
             # Backprop as if integrating up to event time.
             # Does NOT backpropagate through the event time.
             event_mode = ctx.event_mode
             if event_mode:
                  t, z, z_now, y_before, event_t, *adjoint_params = ctx.saved_tensors
                  _t = t
                  t = torch.cat([t[0].reshape(-1), event_t.reshape(-1)])
                  grad_z_l = grad_z_l[1]
             else:
                  t, z, z_now, y_before, *adjoint_params = ctx.saved_tensors
                  grad_z_l = grad_z_l[0]
                     

             grad_z_l=grad_z_l*((z.size()[0]-1)/(t[-1]-t[0]))
             grad_z_l[0]=grad_z_l[0]*2
             grad_z_l[-1]=grad_z_l[-1]*2
                    
             node_params=adjoint_params[0:node_par_len]
             obs_params=adjoint_params[node_par_len:(node_par_len + obs_par_len)]
                
             
                
             states=[z[-1], grad_z_l[-1],torch.zeros(z.size()[1],z.size()[-1])]  #z, grad_z_l, adjoint=lambda
             save_adjoint=torch.zeros(len(t),z.size()[1],z.size()[-1])   
              
              
             def adjoint_ode(time,states): 
                with torch.enable_grad():
                    z_=states[0].detach().requires_grad_(True)
                    adjoint=states[2].detach()
                    func_eval=node_func(time,z_)
                    
                    ans=torch.autograd.grad(func_eval, z_, adjoint,retain_graph=True,allow_unused=True)  #the gradients get multiplied with the adjoint(argument 3)
                    d_adj= -ans[0]-states[1]
                    return [func_eval,states[1],d_adj]

              
             for i in range(len(t) - 1, 0, -1):
                states = odeint(
                    adjoint_ode, tuple(states),
                    t[i - 1:i + 1].flip(0),
                    rtol=adjoint_rtol, atol=adjoint_atol, method=adjoint_method, options=adjoint_options
                ) 
                states = [a[1] for a in states] #states is ode at t_i+1 at index 0 and t_i at index 1, so we need index 1
                states[0] = z[i - 1]  # update to use our forward-pass estimate of the state
                states[1] = grad_z_l[i - 1]  # update to use our forward-pass estimate of the state
                save_adjoint[i-1]=states[2]
                  
             
             
             states=[z[0], save_adjoint[0]]  #z, adjoint
             states.extend([torch.zeros_like(param) for param in node_params]) # L_theta
             #save_function_values=(len(t),[torch.zeros_like(param) for param in node_params])

             
             def L_ode(time,states):
                  with torch.enable_grad():
                      z=states[0]
                      adjoint=states[1].detach()
                      func_eval=node_func(time,z)
                     
                      ans=torch.autograd.grad(func_eval, node_params, adjoint, retain_graph=True, allow_unused=True) #the gradients get multiplied with the adjoint(argument 3), sum over N is automatically included by treating func_eval and adjoint as vectors
                      return [states[0],states[1],*ans]

             for i in range(1, len(t), 1):
               states = odeint(
                   L_ode, tuple(states),
                   t[i - 1:i + 1],
                   rtol=adjoint_rtol, atol=adjoint_atol, method=adjoint_method, options=adjoint_options
               ) 
               states = [a[1] for a in states]  #states is ode at t_i-1 at index 0 and t_i at index 1, so we need index 1
               states[0] = z[i]
               states[1] = save_adjoint[i]
               
             
             grad_L_theta=tuple(states[2:])
  
             
             with torch.enable_grad():
                 func_eval=obs_func(y_before)
                 if len(obs_params) != 0:
                     grad_L_omega=torch.autograd.grad(func_eval, obs_params, save_adjoint[0], retain_graph=True, allow_unused=True)
             
             
             if len(obs_params) != 0:
                 grad_params= grad_L_theta + grad_L_omega
             else:
                 grad_params=grad_L_theta
             return (None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, *grad_params)


def my_odeint_adjoint(node_func, obs_func, t, y_obs, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None,
                    adjoint_rtol=None, adjoint_atol=None, adjoint_method=None, adjoint_options=None, adjoint_params=None, node_par_len=None, obs_par_len=None):

    #COPY FROM ORIGINAL SCRIPT, NOT EVERYTHING IS ADJUSTED, SO MAKE SURE TO USE CORRECT INPUTS
    
    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    
    #TODO: Adjust warning 
    
    # if adjoint_params is None and not isinstance(func, nn.Module):
    #     raise ValueError('func must be an instance of nn.Module to specify the adjoint parameters; alternatively they '
    #                      'can be specified explicitly via the `adjoint_params` argument. If there are no parameters '
    #                      'then it is allowable to set `adjoint_params=()`.')

    # Must come before _check_inputs as we don't want to use normalised input (in particular any changes to options)
    if adjoint_rtol is None:
        adjoint_rtol = rtol
    if adjoint_atol is None:
        adjoint_atol = atol
    if adjoint_method is None:
        adjoint_method = method

    if adjoint_method != method and options is not None and adjoint_options is None:
        raise ValueError("If `adjoint_method != method` then we cannot infer `adjoint_options` from `options`. So as "
                         "`options` has been passed then `adjoint_options` must be passed as well.")

    if adjoint_options is None:
        adjoint_options = {k: v for k, v in options.items() if k != "norm"} if options is not None else {}
    else:
        # Avoid in-place modifying a user-specified dict.
        adjoint_options = adjoint_options.copy()

    if adjoint_params is None:
        adjoint_params = tuple(find_parameters([node_func,obs_func]))
    else:
        adjoint_params = tuple(adjoint_params)  # in case adjoint_params is a generator.

    # Filter params that don't require gradients.
    oldlen_ = len(adjoint_params)
    adjoint_params = tuple(p for p in adjoint_params if p.requires_grad)
    if len(adjoint_params) != oldlen_:
        # Some params were excluded.
        # Issue a warning if a user-specified norm is specified.
        if 'norm' in adjoint_options and callable(adjoint_options['norm']):
            warnings.warn("An adjoint parameter was passed without requiring gradient. For efficiency this will be "
                          "excluded from the adjoint pass, and will not appear as a tensor in the adjoint norm.")

    # Convert to flattened state.
    shapes, node_func, obs_func, y_obs, t, rtol, atol, method, options, event_fn, decreasing_time = check_inputs_adjoint(node_func, obs_func, y_obs, t, rtol, atol, method, options, event_fn, SOLVERS)

#TODO
    # Handle the adjoint norm function.
    # state_norm = options["norm"]
    # handle_adjoint_norm_(adjoint_options, shapes, state_norm)

    ans = OdeintAdjointMethod.apply(shapes, node_func, obs_func, t, y_obs, rtol, atol, method, options, event_fn, adjoint_rtol, adjoint_atol,
                                    adjoint_method, adjoint_options, t.requires_grad, node_par_len, obs_par_len, *adjoint_params)

    if event_fn is None:
        solution = ans
    else:
        event_t, solution = ans
        event_t = event_t.to(t)
        if decreasing_time:
            event_t = -event_t

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)

    if event_fn is None:
        return solution
    else:
        return event_t, solution


def find_parameters(module):

    assert isinstance(module, nn.Module)

    # If called within DataParallel, parameters won't appear in module.parameters().
    if getattr(module, '_is_replica', False):

        def find_tensor_attributes(module):
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v) and v.requires_grad]
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:
        return list(module.parameters())


def handle_adjoint_norm_(adjoint_options, shapes, state_norm):
    """In-place modifies the adjoint options to choose or wrap the norm function."""

    # This is the default adjoint norm on the backward pass: a mixed norm over the tuple of inputs.
    def default_adjoint_norm(tensor_tuple):
        t, y, adj_y, *adj_params = tensor_tuple
        # (If the state is actually a flattened tuple then this will be unpacked again in state_norm.)
        return max(t.abs(), state_norm(y), state_norm(adj_y), _mixed_norm(adj_params))

    if "norm" not in adjoint_options:
        # `adjoint_options` was not explicitly specified by the user. Use the default norm.
        adjoint_options["norm"] = default_adjoint_norm
    else:
        # `adjoint_options` was explicitly specified by the user...
        try:
            adjoint_norm = adjoint_options['norm']
        except KeyError:
            # ...but they did not specify the norm argument. Back to plan A: use the default norm.
            adjoint_options['norm'] = default_adjoint_norm
        else:
            # ...and they did specify the norm argument.
            if adjoint_norm == 'seminorm':
                # They told us they want to use seminorms. Slight modification to plan A: use the default norm,
                # but ignore the parameter state
                def adjoint_seminorm(tensor_tuple):
                    t, y, adj_y, *adj_params = tensor_tuple
                    # (If the state is actually a flattened tuple then this will be unpacked again in state_norm.)
                    return max(t.abs(), state_norm(y), state_norm(adj_y))
                adjoint_options['norm'] = adjoint_seminorm
            else:
                # And they're using their own custom norm.
                if shapes is None:
                    # The state on the forward pass was a tensor, not a tuple. We don't need to do anything, they're
                    # already going to get given the full adjoint state as (t, y, adj_y, adj_params)
                    pass  # this branch included for clarity
                else:
                    # This is the bit that is tuple/tensor abstraction-breaking, because the odeint machinery
                    # doesn't know about the tupled nature of the forward state. We need to tell the user's adjoint
                    # norm about that ourselves.

                    def _adjoint_norm(tensor_tuple):
                        t, y, adj_y, *adj_params = tensor_tuple
                        y = _flat_to_shape(y, (), shapes)
                        adj_y = _flat_to_shape(adj_y, (), shapes)
                        return adjoint_norm((t, *y, *adj_y, *adj_params))
                    adjoint_options['norm'] = _adjoint_norm