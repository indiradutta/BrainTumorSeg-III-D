def train(m,loader,opt,epochs):
    losses = []
    dsc = []
    isc = []
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    m = m.cuda()
    m.train()

    for e in tqdm(range(epochs)):        
        for _,(x,y) in enumerate(loader):
            for i in range(len(x)): 
                opt.zero_grad()

                x[i],y[i] = x[i].float(),y[i].float()
                x[i],y[i] = x[i].cuda(),y[i].cuda()

                out = m(x[i])
                
                loss1 = dice_loss(out,y[i].detach(),multiclass=True)
                loss1.backward()

                opt.step()

        if e%5 == 0:
            plt.imshow(y[i][0,2,:,:,33].cpu().detach())
            plt.show()
            plt.imshow(out[0,2,:,:,33].cpu().detach())
            plt.show()
       
        losses.append(loss1.item())             
        dsc.append(1-loss1.item())

        print('Epoch: ',str(e+1),'Dice Loss: ',str(loss1.item()),'Dice Score: ',str(1-loss1.item()))

        if e%50 == 0:
            model_data = {'model': m.state_dict(),'optimizer': opt.state_dict(),'loss': losses,'dice': dsc}
            torch.save(model_data,'ar-unet3d'+str(e+1)+'.pth')

    model_data = {'model': m.state_dict(),'optimizer': opt.state_dict(),'loss': losses,'dice': dsc}

    plot(losses,dsc)

    torch.save(model_data,'arunet3d.pth')
    print('done :)')
