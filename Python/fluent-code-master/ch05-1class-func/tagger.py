# BEGIN TAG_FUNC
def tag(name, *content, cls=None, **attrs):
    """Generate one or more HTML tags"""
    if cls is not None:
        attrs['class'] = cls
    if attrs:
        attr_str = ''.join(' %s="%s"' % (attr, value)
                           for attr, value
                           in sorted(attrs.items()))
    else:
        attr_str = ''
    if content:
        return '\n'.join('<%s%s>%s</%s>' %
                         (name, attr_str, c, name) for c in content)
    else:
        return '<%s%s />' % (name, attr_str)
# END TAG_FUNC

# BEGIN TAG_DEMO
print('tag("br") = {0}'.format(tag('br')))
print()

print('tag("p", "hello") = {0}'.format(tag('p', 'hello')))
print()

print('tag("p", "hello", "world") = {0}'.format(tag('p', 'hello', 'world')))
print()

print('tag("p", "hello", id=33) = {0}'.format(tag('p', 'hello', id=33)))
print()

print('tag("p", "hello", cls="sidebar") = {0}'.format(tag('p', 'hello', cls='sidebar')))
print()

print('tag(content="testing", name="img") = {0}'.format(tag(content='testing', name="img")))
print()

my_tag = {'name': 'img', 'title': 'Sunset Boulevard', 'src': 'sunset.jpg', 'cls': 'framed'}
print('tag(**my_tag) = \n{0}'.format(tag(**my_tag)))
# END TAG_DEMO

