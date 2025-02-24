#https://www.youtube.com/watch?v=tDv7lrklaQE    MINI MAX
#https://www.youtube.com/watch?v=9D1hVGumxCo    Alpha Beta 

MAX,MIN=1000,-1000

def minmax(depth,nodeIndex,maximizing_player,values,alpha,beta):
    if depth==4:
        return values[nodeIndex]
    if maximizing_player:
        best=MIN
        for i in range(0,2):
            val=minmax(depth+1,nodeIndex*2+i,False,values,alpha,beta)
            best=max(best,val)
            alpha=max(alpha,best)
            if beta<=alpha:
                break
        return best
    else:
        best=MAX
        for i in range(0,2):
            val=minmax(depth+1,nodeIndex*2+i,True,values,alpha,beta)
            best=min(best,val)
            beta=min(beta,best)
            if beta<=alpha:
                break
        return best
        


values=[3,4,2,1,7,8,9,10,2,11,1,12,14,9,13,16,1,15]
print("The optimal value is ",minmax(0,0,True,values,MIN,MAX))