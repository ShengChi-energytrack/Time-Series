

def live_plot(data,missing,ground_truth,figsize=(12,4), title=''):
    #clear_output(wait=True)
    plt.figure(figsize=figsize)
    #plt.ylim(-1.5,3.5)
    #plt.xlim(-2,51)
    missing=missing.cpu()
    x= np.arange(data.shape[1])
    colors=["blue","green","red", "yellow"]
    plot_num = 0
    if data.shape[0] > 4:
        plot_num = 4
    else:
        plot_num = data.shape[0]
    for dim in range(plot_num):
        plt.plot(x,data[dim].cpu(),c=colors[dim])
        plt.plot(x,ground_truth[dim].cpu(),alpha=0.5,linestyle="dashed",c=colors[dim])
        plt.scatter(x[missing[dim]],(data[dim][missing[dim]]).cpu(),c=colors[dim])

    line = Line2D([], [], label='ground truth', color='blue', linestyle="dashed")
    line2 = Line2D([], [], label='reconstruction', color='blue')
    line3 = Line2D([], [], label='missing points', color='blue',marker="o")
    
    plt.legend(handles=[line,line2,line3], numpoints=1,loc=3)
    plt.title("Epoch {}".format(int(title)+1))
#     plt.grid(True)
    plt.xlabel('axis x')
    plt.ylabel('axis y')
    plt.show()
    
def split_data(data, mask, delta):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    train_indices= indices[:int(data.shape[0]*0.8)]
    test_indices = indices[int(data.shape[0]*0.8):]

    data_train = data[train_indices]
    mask_train = mask[train_indices]
    delta_train = delta[train_indices]
    data_test = data[test_indices]
    mask_test = mask[test_indices]
    delta_test = delta[test_indices]
    return data_train,mask_train,delta_train,data_test,mask_test,delta_test

def permute2st(v, ndim_en=1):
    """
    Permute last ndim_en of an array v to the first
    :type v: torch.Tensor
    :type ndim_en: int
    :rtype: torch.Tensor
    """
    nd = v.ndimension()
    return v.permute([*range(-ndim_en, 0)] + [*range(nd - ndim_en)])

def permute2en(v, ndim_st=1):
    """
    Permute first ndim_en of an array v to the last
    :type v: torch.Tensor
    :type ndim_st: int
    :rtype: torch.Tensor
    """
    nd = v.ndimension()
    return v.permute([*range(ndim_st, nd)] + [*range(ndim_st)])

def unblock_diag(m, n=None, size_block=None):
    """
    The inverse of block_diag(). Not vectorized yet.
    :param m: block diagonal matrix
    :param n: int. Number of blocks
    :size_block: torch.Size. Size of a block.
    :return: tensor unblocked such that the last sizes are [n] + size_block
    """
    # not vectorized yet
    if size_block is None:
        size_block = torch.Size(torch.tensor(m.shape[-2:]) // n)
    elif n is None:
        n = m.shape[-2] // torch.tensor(size_block[0])
        assert n == m.shape[-1] // torch.tensor(size_block[1])
        
    m = permute2st(m, 2)

    res = torch.zeros(torch.Size([n]) + size_block + m.shape[2:])
    for i_block in range(n):
        st_row = size_block[0] * i_block
        en_row = size_block[0] * (i_block + 1)
        st_col = size_block[1] * i_block
        en_col = size_block[1] * (i_block + 1)
        res[i_block,:] = m[st_row:en_row, st_col:en_col, :]

    return permute2en(res, 3)

def block_diag(m):
    """
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n).unsqueeze(-2), d - 3, 1)
    return (m2 * eye).reshape(
        siz0 + torch.Size(torch.tensor(siz1) * n)
    )

def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))

def ReLU(x):
    return x * (x > 0)

def lin_interpolation(x_in):
    x = x_in.clone()
    for sample in range(x.shape[0]): # for every sample
        for stream in range(x.shape[1]): # for every stream
            s = pd.Series(x[sample,stream])
            s = s.interpolate().bfill().ffill() # linear interpolation, then backward and forward fill for first and last value
            x[sample,stream] = torch.tensor(s.values) # replace in original tensor
    return x