def remove_comments(string):
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    # first group captures quoted strings (double or single)
    # second group captures comments (//single-line or /* multi-line */)
    regex = re.compile(pattern, re.MULTILINE|re.DOTALL)
    def _replacer(match):
        # if the 2nd group (capturing comments) is not None,
        # it means we have captured a non-quoted (real) comment string.
        if match.group(2) is not None:
            return "" # so we will return empty to remove the comment
        else: # otherwise, we will return the 1st group
            return match.group(1) # captured quoted-string
    return regex.sub(_replacer, string)

with open("temp_tokens_2.csv", "r") as f1:
    lines = f1.readlines()
    for sent in lines:
        sent = sent.replace("\n", "")
        sent = remove_comments(string)

        with open("temp_tokens_3.csv", "w") as f2:
            f2.write(sent)
            f2.write("\n")