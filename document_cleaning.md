Word_File_Cleaning applies extensive **formatting, cleanup, and citation standardisation** to a Word document.

The primary objective of the script is to **normalise appearance** and **correct common stylistic inconsistencies**, particularly in citations, spelling, and metadata, based on an identified jurisdiction.

---

## Summary of Fixes and Transformations

### 1. Document Style and General Cleanup

| Action | Description |
| :--- | :--- |
| **Apply Normal Style** | Sets the default ‘Normal’ paragraph style to **Calibri, 12pt** font, with **0pt spacing before** and **6pt spacing after** paragraphs. |
| **Remove Empty or Redundant Paragraphs** | **Deletes** paragraphs that are **empty** (containing only whitespace) or match the dynamically generated `HEADER_TEXT`. |
| **Normalize Punctuation Spacing** | **Removes extraneous space** before punctuation marks such as `.`, `,`, `:`, `;`, `!`, `?`, `)`, and `]`. |
| **Normalize General Whitespace** | Replaces **double internal newlines** (`\n\n`) with a single space and standardises **non-breaking or irregular double spaces** to a single standard space. |
| **De-Americanise Spelling** | Converts **American English spellings** (using a global `AMERICAN_TO_BRITISH` mapping) to **British English equivalents** (e.g., *organize* → *organise*, *labor* → *labour*). |
| **Remove Word Count Annotations** | **Purges any bracketed expression**—whether in parentheses `(...)`, square brackets `[...]`, or curly braces `{...}`—that contains the phrase **“word count”** (case-insensitive), including the brackets themselves. This prevents metadata like `(word count: 250)` from appearing in final outputs. |

---

### 2. Citation and Page Reference Standardisation

The code applies a sequence of targeted transformations to content within parentheses, focusing on academic citation conventions:

| Action | Utility Function | Description |
| :--- | :--- | :--- |
| **Merge Adjacent Citations** | `merge_adjacent_citations_in_paragraph` | Combines consecutive parenthetical citations (e.g., `(p. 10). (p. 15)`) into a single group (e.g., `(p. 10, p. 15)`), **provided both contain page prefixes** (`p.` or `pp.`) and are separated only by whitespace or a full stop. |
| **Fix Citation Separator** | `replace_semicolon_in_citations` | Replaces **semicolons (`;`) with commas (`,`)** *inside* parentheses, but **only if** a page reference (`p.` or `pp.`) is present. |
| **Expand Implicit Prefixes** | `expand_implicit_page_prefixes` | Expands shorthand page lists to include the prefix on each number (e.g., changes `(p. 6, 7)` to **`(p. 6, p. 7)`**). |
| **Normalise Page Range Prefix** | `normalize_page_citations_in_parentheses` | Corrects single-page prefix (`p.`) to the **plural form (`pp.`)** when citing a **page range** (e.g., changes `(p. 6–7)` to **`(pp. 6–7)`**). |
| **Normalise Citation Whitespace** | `normalize_citation_whitespace` | Ensures **exactly one space** between the page prefix and the number (e.g., changes `(pp.5)` or `(pp.  5)` to **`(pp. 5)`**). |
| **Deduplicate Page References** | `deduplicate_page_citations` | Removes **repeated identical page references** within a single parenthetical citation (e.g., changes `(p. 15, p. 15)` to **`(p. 15)`**). |

---

### 3. Header Setup and Subheading Formatting

| Action | Description |
| :--- | :--- |
| **First Page Header Configuration** | Configures the document to use a **different first-page header**, ensuring branding or titles appear only on page one. |
| **Header Cleanup & Creation** | **Clears all existing content** from both the first-page and primary headers, then inserts a **new paragraph** in the first-page header containing the dynamic `HEADER_TEXT` (e.g., *“England Accountability – RQ aligned summary”*). |
| **Header Formatting** | Styles the first-page header text as **Calibri, 14pt, bold, and dark blue** for visual consistency and professional appearance. |
| **Subheading Identification** | Uses heuristics—such as short length, absence of terminal punctuation, and exclusion of common non-heading phrases—to **identify candidate subheadings** in the body text. |
| **Subheading Standardisation** | Replaces or formats identified subheadings to match a **canonical list** of expected headings. Text matching **Research Question (RQ) titles** is styled in **dark blue**, while standard subheadings appear in **black**, both in **bold Calibri 12pt**. |

---

This comprehensive pipeline ensures that heterogeneous academic drafts are transformed into **consistent, professionally formatted, UK-conforming policy summaries**, suitable for dissemination by research or policy organisations.
