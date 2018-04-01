function knapSack(W, wt, val, n)

    dp = torch.Tensor(n+1, W+1)
 

    for i=1, n+1 do

        for w=1, W+1 do

            if i==1 or w==1 then

                dp[i][w] = 0

            elseif wt[i-1] <= w then

                dp[i][w] = math.max(val[i-1] + dp[i-1][w-wt[i-1]],  dp[i-1][w])

            else

                dp[i][w] = dp[i-1][w]
            end
        end
    end
 
    return dp[n+1][W+1]
end
 

val = {60, 100, 120}
wt = {10, 20, 30}

W = 50
n = #val

print(knapSack(W, wt, val, n))