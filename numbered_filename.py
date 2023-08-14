"""Provide a function for creating sequentially incremented filenames
based upon a simple template in which an asterisk is replaced with a
number. The filesystem is checked for existing files that match the
template and the returned filename's sequence number is always one
greater than the maximum found. 
"""

import glob
if __debug__:
    import os

def numbered_filename(template :str ='', width :int =3) -> str:
    """Return the next filename in an incrementing sequence by adding
    one to the current largest number in existing filenames.

    template :str: a string with an asterisk in it representing where
                   the numbers are placed. ('foo-*.txt').

       width :int: optional minimum number of digits to zero-pad the
                   sequence to. Defaults to 3 ('000', '001', '002', ...)

    Example usage:

        from numbered_filename import numbered_filename
        newfile = numbered_filename('foo-*.txt')
        with open(newfile, 'w') as outfile:
            outfile.write("Bob's your uncle!")

    Given a filename template with an asterisk in it, such as
    'foo-*.txt', returns the same filename with the asterisk replaced
    with the next number in the sequence, such as 'foo-007.txt'. If no
    prior file exists, numbering starts at zero ('foo-000.txt').

    The number will be left-padded with zeroes to contain at least
    three digits, unless the optional 'width' argument is given.
    Zero-padding can be disabled with 'width=1'. For example,
    'numbered_filename("hackerb*", width=1)' might return 'hackerb9'.
    Note that 'width' is a minimum and more digits will be used if
    necessary (e.g., 'foo-1000.txt').

    Regardless of the 'width' setting, existing filenames need not be
    zero-padded to be recognized. For example, if a directory has the
    file 'foo-6.txt', the next filename will be 'foo-007.txt'.

    This routine always return the next higher number after any
    existing file, even if a lower number is available. For example,
    in a directory containing only 'foo-099.txt', the next file would
    be 'foo-100.txt', despite 'foo-000' through '-098.txt' being free.

    Peculiar Circumstances:

    * If the template is the empty string (''), then the output will
      simply be a sequence number ('007').

    * If the template contains no asterisks ('foo'), then the number
      is appended to the end of the filename ('foo007').

    * If more than one asterisk is used ('*NSYNC*.txt'), then only the
      rightmost asterisk is replaced with a number ('*NSYNC007.txt').
      All others asterisks are kept as literal '*' in the filename.

    * Specifying 'width=0' will use the maximum width in any existing
      file (e.g., 'foo-00008.txt' -> width=5).

    CAVEAT: While the code attempts to return an unused filename, it
    is not guaranteed as there is a fairly obvious race condition.
    To avoid it, processes writing to the same directory concurrently
    must not use the same template. Do not use this to create temp
    files in a directory where an adversary may have write access,
    such as /tmp -- instead use 'mkstemp'.

    """

    if not isinstance(template, str):
        raise TypeError("numbered_filename() requires a string as a template, such as foo-*.txt")

    if not isinstance(width, int):
        raise TypeError("numbered_filename() requires 'width' to be an int or omitted")

    (filename, asterisk, extension) =  template.rpartition('*')
    if not asterisk:
        (filename, extension) = (extension, filename)
        template=f'{filename}*'

    try:
        # Existing file numbers and their string lengths
        nw = [( int(f), len(f) )
              for f in [g.lstrip(filename).rstrip(extension)
                        for g in glob.glob(template)]
              if f.isdigit()]
        num = max(nw)[0]
        if width == 0:
            width = max([w for (dummy, w) in nw])
    except (IndexError, ValueError):
        num = -1

    num = num + 1
    spec = f'0>{width}'
    numstr = format(num, spec)

    result = filename + numstr + extension


    if __debug__ and os.path.exists(result):
        raise AssertionError(f'Error: "{result}" already exists. Race condition or bug?')

    return result
