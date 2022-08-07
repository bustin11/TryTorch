import numpy as np


def GreedySearch(SymbolSets, y_probs):
    """Greedy Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    Returns
    ------
    forward_path: str
                the corresponding compressed symbol sequence i.e. without blanks
                or repeated symbols.

    forward_prob: scalar (float)
                the forward probability of the greedy path

    """
    # Follow the pseudocode from lecture to complete greedy search :-)

    

    BLANK = '_'
    SymbolSets.insert(0, BLANK)

    greedy_str = BLANK
    forward_prob = 1.0

    # TODO: does not work with batch size > 1
    rs = np.argmax(y_probs, axis=0)
    for i, rb in enumerate(rs): # seq len
        for j, r in enumerate(rb): # batch size
            if greedy_str[-1] != SymbolSets[r]:
                greedy_str += SymbolSets[r]
            forward_prob *= y_probs[r][i][j]
    
    forward_path = greedy_str.replace(BLANK, '')
    return (forward_path, forward_prob)


##############################################################################


def InitializePaths(SymbolSet, y):

    InitialBlankPathScore = {}

    # First push the blank into a path-ending-with-blank stack. No symbol has been invoked yet
    path = '' # a path is a string
    blank = 0
    InitialBlankPathScore[path] = y[blank] # Score of blank at t=1

    # Push rest of the symbols into a path-ending-with-symbol stack
    InitialPathScore = {}
    for i, c in enumerate(SymbolSet): # This is the entire symbol set, without the blank
        path = c
        InitialPathScore[path] = y[i+1] # Score of symbol c at t=1
    return InitialBlankPathScore, InitialPathScore





# Global PathScore, BlankPathScore
def ExtendWithSymbol(SymbolSet, y, BlankPathScore, PathScore):

    UpdatedPathScore = {}

    # First extend the paths terminating in blanks. This will always create a new sequence
    for path in BlankPathScore:
        for i, c in enumerate(SymbolSet): # SymbolSet does not include blanks
            newpath = path + c # Concatenation string
            UpdatedPathScore[newpath] = BlankPathScore[path] * y[i+1]

    # Next work on paths with terminal symbols
    for path in PathScore:
        # Extend the path with every symbol other than blank
        for i, c in enumerate(SymbolSet): # SymbolSet does not include blanks

            newpath = path + c
            if c == path[-1]: # Horizontal transitions don’t extend the sequence
                newpath = path

            if newpath in UpdatedPathScore: # Already in list, merge paths
                UpdatedPathScore[newpath] += PathScore[path] * y[i+1]
            else: # Create new path
                UpdatedPathScore[newpath] = PathScore[path] * y[i+1]

    return UpdatedPathScore




# Global PathScore, BlankPathScore
def ExtendWithBlank(y, BlankPathScore, PathScore):

    UpdatedBlankPathScore = {}
    blank = 0

    # First work on paths with terminal blanks
    #(This represents transitions along horizontal trellis edges for blanks)
    for path in BlankPathScore:
        # Repeating a blank doesn’t change the symbol sequence
        UpdatedBlankPathScore[path] = BlankPathScore[path] * y[blank]

    # Then extend paths with terminal symbols by blanks
    for path in PathScore:
        # If there is already an equivalent string in UpdatesPathsWithTerminalBlank
        # simply add the score. If not create a new entry
        if path in UpdatedBlankPathScore: 
            UpdatedBlankPathScore[path] += PathScore[path] * y[blank]
        else:
            UpdatedBlankPathScore[path] = PathScore[path] * y[blank]

    return UpdatedBlankPathScore




# Global PathScore, BlankPathScore
def Prune(BlankPathScore, PathScore, BeamWidth):

    PrunedBlankPathScore = {}
    PrunedPathScore = {}

    # Sort and find cutoff score that retains exactly BeamWidth paths
    scorelist = [v for v in BlankPathScore.values()] + [v for v in PathScore.values()]
    scorelist.sort(reverse=True) # In decreasing order

    cutoff = scorelist[-1] # highest score # highest scor e
    if BeamWidth < len(scorelist):
        cutoff = scorelist[BeamWidth-1]

    for p in BlankPathScore:
        if BlankPathScore[p] >= cutoff:
            PrunedBlankPathScore[p] = BlankPathScore[p]

    for p in PathScore:
        if PathScore[p] >= cutoff:
            PrunedPathScore[p] = PathScore[p]

    return PrunedBlankPathScore, PrunedPathScore


def MergeIdenticalPaths(BlankPathScore, PathScore):

    # All paths with terminal symbols will remain
    FinalPathScore = PathScore

    # Paths with terminal blanks will contribute scores to existing identical paths from
    # PathsWithTerminalSymbol if present, or be included in the final set, otherwise
    for p in BlankPathScore:
        if p in FinalPathScore:
            FinalPathScore[p] += BlankPathScore[p]
        else:
            FinalPathScore[p] = BlankPathScore[p]

    return FinalPathScore



def BeamSearch(SymbolSets, y_probs, BeamWidth):
    """Beam Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    BeamWidth: int
                Width of the beam.

    Return
    ------
    bestPath: str
            the symbol sequence with the best path score (forward probability)

    mergedPathScores: dictionary
                        all the final merged paths with their scores.

    """
    # Follow the pseudocode from lecture to complete beam search :-)
#    import pdb
#    pdb.set_trace()

    SymbolSet = SymbolSets
    y = y_probs
    
    PathScore, BlankPathScore = [], []
    # First time instant: Initialize paths with each of the symbols,
    # including blank, using score at time t=1
    NewBlankPathScore, NewPathScore = InitializePaths(SymbolSet, y[:,0,0])

    T = y_probs.shape[1]
    # Subsequent time steps
    for t in range(1, T):

        # Prune the collection down to the BeamWidth
        BlankPathScore, PathScore = Prune(NewBlankPathScore, NewPathScore, BeamWidth)

        # First extend paths by a blank
        NewBlankPathScore = ExtendWithBlank(y[:,t,0], BlankPathScore, PathScore)

        # Next extend paths by a symbol
        NewPathScore = ExtendWithSymbol(SymbolSet, y[:,t,0], BlankPathScore, PathScore)

    # Merge identical paths differing only by the final blank
#    pdb.set_trace()
    MergedPathScores = MergeIdenticalPaths(NewBlankPathScore, NewPathScore)

    # Pick best path
    BestValue = 0
    BestPath = ''
    for k,v in MergedPathScores.items():
        if v > BestValue:
            BestValue = v 
            BestPath = k # Find the path with the best score

    bestPath = BestPath
#    pdb.set_trace()
    return (bestPath, MergedPathScores)






