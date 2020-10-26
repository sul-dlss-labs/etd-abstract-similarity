## Abstracts DataFrame
For all of the world-accessible theses and dissertations, we queried Stanford's
Digital Repository with value in *druids* column and retrieved the abstract
contained in the MODS XML data-stream.  Each abstract was then processed to
remove stop words and punctuation marks, converted to lower case, and stored
in the *abstracts_cleaned* column. Finally, the department name was retrieved
from the  MODS and added as the *department* column.
