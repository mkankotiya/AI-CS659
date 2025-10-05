import heapq, re

def normalize(s):
    return re.sub(r'[^a-z0-9\s]', '', s.lower()).strip()

def sentences_from_text(text):
    return [normalize(p) for p in re.split(r'[.!?]', text) if p.strip()]

def levenshtein(a, b):
    la, lb = len(a), len(b)
    dp = list(range(lb+1))
    for i in range(1, la+1):
        prev = dp[0]; dp[0] = i
        for j in range(1, lb+1):
            tmp = dp[j]
            cost = 0 if a[i-1]==b[j-1] else 1
            dp[j] = min(dp[j]+1, dp[j-1]+1, prev+cost)
            prev = tmp
    return dp[lb]

def a_star_align(s1, s2, skip_cost=5):
    n, m = len(s1), len(s2)
    start, goal = (0,0), (n,m)
    open_heap = [(0,0,start,None)]
    g_cost, parents, visited = {start:0}, {}, set()
    while open_heap:
        f,g,state,parent = heapq.heappop(open_heap)
        if state in visited: continue
        visited.add(state); parents[state] = parent
        i,j = state
        if state==goal: break
        if i<n and j<m:
            cost = levenshtein(s1[i], s2[j])
            nei=(i+1,j+1); ng=g+cost
            if nei not in g_cost or ng<g_cost[nei]:
                g_cost[nei]=ng; heapq.heappush(open_heap,(ng,ng,nei,(state,'align')))
        if i<n:
            nei=(i+1,j); ng=g+skip_cost
            if nei not in g_cost or ng<g_cost[nei]:
                g_cost[nei]=ng; heapq.heappush(open_heap,(ng,ng,nei,(state,'skip1')))
        if j<m:
            nei=(i,j+1); ng=g+skip_cost
            if nei not in g_cost or ng<g_cost[nei]:
                g_cost[nei]=ng; heapq.heappush(open_heap,(ng,ng,nei,(state,'skip2')))
    return g_cost.get(goal,None)

if __name__ == "__main__":
    text1 = "Machine learning is a field of AI. It studies algorithms that learn from data."
    text2 = "Machine learning belongs to AI. It studies methods that learn from datasets."
    s1, s2 = sentences_from_text(text1), sentences_from_text(text2)
    cost = a_star_align(s1,s2)
    print("Total alignment cost:", cost)
