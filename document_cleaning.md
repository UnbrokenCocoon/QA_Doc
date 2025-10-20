The provided code, specifically the `clean_word_document` function and its utilities, applies extensive **formatting, cleanup, and citation standardization** to a Word file.

The main goal of the script is to normalize the appearance and correct common stylistic errors, particularly concerning citations and spelling, in a document based on an identified jurisdiction.

---

## Summary of Fixes and Transformations

### 1. Document Style and Cleanup

| Action | Description |
| :--- | :--- |
| **Apply Normal Style** | Sets the default 'Normal' paragraph style to **Calibri, 12pt** font, with **0pt spacing before** and **6pt spacing after** paragraphs. |
| **Remove Empty/Header Paragraphs** | **Deletes** paragraphs that are **empty** (only whitespace) or contain the specific dynamically-generated `HEADER_TEXT`. |
| **Normalize Punctuation Spacing** | **Removes space** before punctuation marks like `.`, `,`, `:`, `;`, `!`, `?`, `)`, and `]`. |
| **Normalize General Whitespace** | Replaces **double internal newlines** (`\n\n`) with a single space and replaces **non-standard double spaces** (` "  "`) with a single standard space. |
| **De-Americanize Spelling** | Converts **American English spellings** (using a global `AMERICAN_TO_BRITISH` map) to **British English**. |

---

### 2. Citation and Page Reference Standardization

The code applies multiple sequential fixes targeting content within parentheses (citations):

| Action | Utility Function | Description |
| :--- | :--- | :--- |
| **Merge Adjacent Citations** | `merge_adjacent_citations_in_paragraph` | Combines consecutive parenthetical citations (e.g., `(p. X). (p. Y)`) into a single group (e.g., `(p. X, p. Y)`), provided both contain page prefixes (`p.` or `pp.`). |
| **Fix Citation Separator**| `replace_semicolon_in_citations` | Replaces **semicolons (`;`) with commas (`,`)** *inside* parentheses, but **only if** a page reference (`p.` or `pp.`) is present. |
| **Expand Implicit Prefixes** | `expand_implicit_page_prefixes` | Expands shorthand page ranges to include the prefix on each page number (e.g., changes `(p. 6, 7)` to **`(p. 6, p. 7)`**). |
| **Normalize Page Range Prefix** | `normalize_page_citations_in_parentheses` | Corrects single-page prefix (`p.`) to **plural prefix (`pp.`)** when the citation refers to a **page range** (e.g., changes `(p. 6-7)` to **`(pp. 6-7)`**). |
| **Normalize Citation Whitespace**| `normalize_citation_whitespace` | Ensures there is **exactly one space** between the page prefix and the number (e.g., changes `(pp.5)` or `(pp.  5)` to **`(pp. 5)`**). |
| **Deduplicate Page References** | `deduplicate_page_citations` | Removes **repeated identical page references** within a single parenthetical citation (e.g., changes `(p. 15, p. 15)` to **`(p. 15)`**). |

---

### 3. Header Setup and Subheading Formatting

| Action | Description |
| :--- | :--- |
| **First Page Header Setup** | Sets the **first page header** to be different from the rest of the document. |
| **Header Cleanup & Creation** | **Clears all existing content** from both the first page and primary headers. It then adds a **new paragraph** to the first page header containing the dynamic `HEADER_TEXT`. |
| **Header Formatting** | Formats the new first page header text as **Calibri, 14pt, Bold, and Dark Blue**. |
| **Subheading Identification** | Uses heuristics (short length, lack of full stops) and a predefined list to **identify potential subheadings** in the document text. |
| **Subheading Formatting** | **Bolds and colors** any paragraph text that matches the dynamically generated 'RQ Full Titles' (Dark Blue) or the 'Standard Subheadings' list (Black). |
