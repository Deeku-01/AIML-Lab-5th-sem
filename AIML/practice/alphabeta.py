MAX,MIN=1000,-1000

def minimax(depth,nodeIndex,max_player,values,alpha,beta):
    if depth==3:
        return values[nodeIndex]
    if max_player:
        best=MIN
        for i in range(0,2):
            val=minimax(depth+1,nodeIndex*2+i,False,values,alpha,beta)
            best=max(best,val)
            alpha=max(best,alpha)
            if beta<=alpha:
                break
        return best
    else:
        best=MAX
        for i in range(0,2):
            val=minimax(depth+1,nodeIndex*2+i,True,values,alpha,beta)
            best=min(best,val)
            beta=min(best,beta)
            if beta<=alpha:
                    break
        return best


values=[3,5,14,1,8,6,4,7,10,11,13,16,17,8]
print("the Optimal value is",minimax(0,0,True,values,MIN,MAX))
