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
	umls.MRREL
LEFT JOIN umls.MRCONSO c1 ON
	umls.MRREL.AUI1 = c1.AUI
LEFT JOIN umls.MRCONSO c2 ON
	umls.MRREL.AUI2 = c2.AUI
WHERE
	RELA IN ("causative_agent_of", "has_occurrence", "cause_of", "direct_device_of", "focus_of", "finding_method_of", "associated_finding_of", "has_modification")
	AND c2.STR NOT LIKE "% %";