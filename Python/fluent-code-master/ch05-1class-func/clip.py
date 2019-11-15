# BEGIN CLIP
def clip(text, max_len=80):
    '''
    Return text clipped at the last space before or after max_len
    '''
    end = None
    if len(text) > max_len:
        space_before = text.rfind(' ', 0, max_len)
        if space_before >= 0:
            end = space_before
        else:
            space_after = text.rfind(' ', max_len)
            if space_after >= 0:
                end = space_after
    if end is None:  # no spaces were found
        end = len(text)
    return text[:end].rstrip()
# END CLIP

print('clip("banana ", 6) = {0}'.format(clip('banana ', 6)))
print()

print('clip("banana ", 7) = {0}'.format(clip('banana ', 7)))
print()

print('clip("banana ", 5) = {0}'.format(clip('banana ', 5)))
print()

print('clip("banana split", 6) = {0}'.format(clip('banana split', 6)))
print()

print('clip("banana split", 7) = {0}'.format(clip('banana split', 7)))
print()

print('clip("banana split", 10) = {0}'.format(clip('banana split', 10)))
print()

print('clip("banana split", 11) = {0}'.format(clip('banana split', 11)))
print()

print('clip("banana split", 12) = {0}'.format(clip('banana split', 12)))
print()
