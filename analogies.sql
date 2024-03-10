SELECT
	CUI1,
	AUI1,
	c1.STR,
	REL,
	RELA,
	CUI2,
	AUI2,
	c2.STR
FROM
	-- Change the schema name to the one you are using, in this case, the schema is called umls.
	umls.MRREL
LEFT JOIN umls.MRCONSO c1 ON
	umls.MRREL.AUI1 = c1.AUI
LEFT JOIN umls.MRCONSO c2 ON
	umls.MRREL.AUI2 = c2.AUI
WHERE
	RELA IN ("finding_method_of", "associated_with", "has_causative_agent")
	AND c2.STR NOT LIKE "% %";